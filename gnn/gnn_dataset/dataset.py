import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import shutil
from collections import Counter, OrderedDict

import torch_geometric
from torch_geometric.data import Data

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, MultiLabelBinarizer, LabelEncoder


def csr_to_long_tensor(csr_matrix):
    coo = csr_matrix.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.LongTensor(values)
    shape = coo.shape

    return torch.sparse.LongTensor(i, v, torch.Size(shape)).to_dense()


def unique(array):
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]


def flatten_list(t):
    return [item for sublist in t for item in sublist]


def process_data(dataset, df_item, df_inter):
    item_encoder = LabelEncoder()
    scaler = MinMaxScaler()
    encoder = OneHotEncoder(sparse=False)
    seq_encoder = MultiLabelBinarizer(sparse_output=False)

    # factorize items for node ids 0...n for graph
    df_item = df_item.sort_values(by=['item_id:token'], ascending=True)
    df_item.insert(len(df_item.columns) - 1, 'item_id', item_encoder.fit_transform(df_item['item_id:token']))
    #df_item = df_item.drop_duplicates(subset='item_id:token', keep="first")
    df_item = df_item.groupby(list(df_item.columns.values)[:-1], as_index=False).agg({'train_valid_test': lambda x: list(x)})

    # filter out augmented sessions...keep longest session
    df_inter = df_inter.sort_values(by=['session_id_original:token', 'timestamp:float'], ascending=True)
    df_inter = df_inter.drop_duplicates(subset='session_id_original:token', keep="last").reset_index(drop=True)

    df_inter['item_id'] = item_encoder.transform(df_inter['item_id:token'])

    df_inter['item_id_list'] = df_inter['item_id_list:token_seq'].str.split(" ").apply(lambda x: list(map(int, x)))
    df_inter['item_id_list'] = df_inter['item_id_list'].apply(lambda x: item_encoder.transform(x))
    df_inter['item_id_list'] = df_inter['item_id_list'].apply(lambda x: x.tolist()) + df_inter['item_id'].apply(lambda x: [x])

    df_tokens = pd.DataFrame(df_inter['item_id_list'].values.tolist())

    if dataset.item_cat_feature_names:
        encoder.fit(df_item[dataset.item_cat_feature_names].to_numpy())
    if dataset.item_num_feature_names:
        scaler.fit(df_item[dataset.item_num_feature_names].to_numpy())
    if dataset.item_seq_feature_name:
        df_item[dataset.item_seq_feature_name] = df_item[dataset.item_seq_feature_name].apply(
            lambda x: np.array([s for s in x.split(' ')]))
        seq_encoder.fit(df_item[dataset.item_seq_feature_name])

    node_feature_lst = []
    edges_src_lst = []
    edges_dst_lst = []
    train_mask = [False] * df_item.shape[0]
    valid_mask = [False] * df_item.shape[0]
    test_mask = [False] * df_item.shape[0]
    for idx, item in tqdm(df_item.iterrows(), total=df_item.shape[0], miniters=int(df_item.shape[0] / 100)):
        item_sessions = df_inter.loc[((df_tokens == item['item_id']).sum(1).astype(bool))
                                     & (~df_inter['train_valid_test'].isin([1, 2]))]  # only training sessions

        sess_items = None
        if dataset.directed or not dataset.session_based:
            item_pos = [(i, item_list.index(item['item_id'])) for i, item_list in
                        enumerate(item_sessions['item_id_list'].values)]
            pre_pos = [(i, j - 1) for i, j in item_pos if j > 0]
            predecessors = [item_sessions['item_id_list'].values[i][j] for i, j in pre_pos]
            sess_items = np.array(predecessors)
            if len(sess_items) == 0:  # no sessions available for item
                sess_items = np.array([item['item_id']])  # self loop
        else:
            # session based
            sess_items = item_sessions['item_id_list'].values
            if len(sess_items) == 0:  # no sessions available for item
                sess_items = np.array([item['item_id']])  # self loop
            elif len(sess_items) > 1:
                sess_items = np.concatenate(sess_items)
            else:
                sess_items = sess_items[0]

        # item features
        # node features are index based! node 0 has features on index 0
        node_features = []
        if dataset.item_num_feature_names:
            num_node_features = scaler.transform([item[dataset.item_num_feature_names]])
            node_features.append(num_node_features[0])
        if dataset.item_cat_feature_names:
            cat_node_features = encoder.transform([item[dataset.item_cat_feature_names]])
            node_features.append(cat_node_features[0])
        if dataset.item_emb_feature_names:
            emb_features = []
            for emb_feat in dataset.item_emb_feature_names:
                emb_features = emb_features + [float(s) for s in item[emb_feat].split(' ')]
            node_features.append(emb_features)
        if dataset.item_seq_feature_name:
            seq_node_features = seq_encoder.transform([item[dataset.item_seq_feature_name]])
            node_features.append(seq_node_features[0])

        node_features = torch.FloatTensor(np.hstack(node_features).ravel())
        node_feature_lst.append(node_features)

        if 0 in item['train_valid_test']:
            train_mask[idx] = True
        elif 1 in item['train_valid_test']:
            valid_mask[idx] = True
        elif 2 in item['train_valid_test']:
            test_mask[idx] = True

        # create edges
        if dataset.directed:
            if 0 not in item['train_valid_test'] and dataset.inductive:
                source_nodes = np.array(
                    [item['item_id']])
            else:
                source_nodes = sess_items

            target_nodes = np.repeat(item['item_id'], sess_items.shape[0])
        else:
            if 0 not in item['train_valid_test'] and dataset.inductive:
                target_nodes = np.array([item['item_id']])  # skip edge information (inductive setting)...only self loop
            else:
                target_nodes = np.delete(sess_items, np.where(sess_items == item['item_id']))
                target_nodes = np.append(target_nodes, item['item_id']) # implicit self loop

            source_nodes = np.repeat(item['item_id'], target_nodes.shape[0])

        edges_src_lst.extend(source_nodes)
        edges_dst_lst.extend(target_nodes)

    # unique edges and edge weights
    edges_tpl = list(zip(edges_src_lst, edges_dst_lst))
    edges_tpl_srtd = list(map(tuple, map(sorted, edges_tpl)))

    edges_counter = Counter(edges_tpl_srtd)

    keys, edge_weights = edges_counter.keys(), list(edges_counter.values())

    edges_src_lst, edges_dst_lst = zip(*keys)

    # node degrees
    node_degrees = Counter(edges_src_lst + edges_dst_lst)
    node_degrees = OrderedDict(node_degrees)
    max_node_degree = max(list(node_degrees.values()))
    node_degrees = list(node_degrees.values())
    node_degrees_norm = np.array(node_degrees) / max_node_degree

    # normalize edge weights (by outdegree of edge start node)
    edge_weights_norm = np.array([weight / node_degrees[edges_src_lst[i]] for i, weight in enumerate(edge_weights)])

    node_features = torch.cat((torch.stack(node_feature_lst), torch.Tensor(node_degrees_norm).unsqueeze(dim=1)), dim=1)

    if not dataset.directed:
        # undirected graph needs edges for both directions
        edge_weights = torch.from_numpy(np.concatenate((edge_weights_norm, edge_weights_norm))).type(torch.FloatTensor)
        edges_src = torch.from_numpy(np.array(list(edges_src_lst) + list(edges_dst_lst)))
        edges_dst = torch.from_numpy(np.array(list(edges_dst_lst) + list(edges_src_lst)))
    else:
        edge_weights = torch.from_numpy(edge_weights_norm).type(torch.FloatTensor)
        edges_src = torch.from_numpy(np.array(list(edges_src_lst)))
        edges_dst = torch.from_numpy(np.array(list(edges_dst_lst)))

    return node_features, edges_src, edges_dst, edge_weights, item_encoder, (train_mask, valid_mask, test_mask)


class BaseSessionGNNDatasetPyG(torch_geometric.data.InMemoryDataset):
    def __init__(self, dataset_name=None, pre_dataset_path=None,
                 save_dir=None, transform=None, pre_transform=None,
                 inter_names=None, item_names=None, item_num_feature_names=None, item_cat_feature_names=None,
                 item_seq_feature_name=None, item_emb_feature_names=None,
                 second_lvl_session_edges=False, use_chached=False, inductive=True):
        self.dataset_name = dataset_name
        self.use_chached = use_chached
        self.inductive = inductive

        assert pre_dataset_path is not None
        self.pre_dataset_path = pre_dataset_path

        self.second_lvl_session_edges = second_lvl_session_edges
        self.inter_names = inter_names
        self.item_names = item_names
        self.item_num_feature_names = item_num_feature_names
        self.item_cat_feature_names = item_cat_feature_names
        self.item_seq_feature_name = item_seq_feature_name
        self.item_emb_feature_names = item_emb_feature_names

        self.directed = False  # if true: only directed edges between predecessor and item
        self.session_based = True  # if false: only edges between predecessor and item

        assert save_dir, "no save dir"
        save_dir = os.path.join(save_dir, f"{dataset_name}-pyg/")

        if not use_chached:
            shutil.rmtree(save_dir, ignore_errors=True)

        super().__init__(save_dir, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'{self.dataset_name}.dataset']

    def download(self):
        pass

    def process(self):
        df_train_inter = pd.read_csv(self.pre_dataset_path + f'{self.dataset_name}.train.inter', sep='\t', header=0,
                               names=self.inter_names)
        df_valid_inter = pd.read_csv(self.pre_dataset_path + f'{self.dataset_name}.valid.inter', sep='\t', header=0,
                                     names=self.inter_names)
        df_test_inter = pd.read_csv(self.pre_dataset_path + f'{self.dataset_name}.test.inter', sep='\t', header=0,
                                     names=self.inter_names)
        df_train_inter['train_valid_test'] = 0
        df_valid_inter['train_valid_test'] = 1
        df_test_inter['train_valid_test'] = 2
        df_inter = pd.concat([df_train_inter, df_valid_inter, df_test_inter], ignore_index=True, sort=False)

        df_train_item = pd.read_csv(self.pre_dataset_path + f'{self.dataset_name}.train.item', sep='\t', header=0,
                              names=self.item_names)
        df_valid_item = pd.read_csv(self.pre_dataset_path + f'{self.dataset_name}.valid.item', sep='\t', header=0,
                                    names=self.item_names)
        df_test_item = pd.read_csv(self.pre_dataset_path + f'{self.dataset_name}.test.item', sep='\t', header=0,
                                    names=self.item_names)
        df_train_item['train_valid_test'] = 0
        df_valid_item['train_valid_test'] = 1
        df_test_item['train_valid_test'] = 2
        df_item = pd.concat([df_train_item, df_valid_item, df_test_item], ignore_index=True, sort=False)

        node_features, edges_src, edges_dst, edge_weights, encoder, masks = \
            process_data(self, df_item, df_inter)

        np.save(self.pre_dataset_path + f'{self.dataset_name}_item_labels.npy', encoder.classes_)

        edge_index = torch.stack((edges_src, edges_dst)).long()

        data = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weights)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # masks for data split
        data.train_mask = torch.BoolTensor(masks[0])
        data.val_mask = torch.BoolTensor(masks[1])
        data.test_mask = torch.BoolTensor(masks[2])

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])
