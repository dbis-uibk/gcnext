import torch
from torch import nn

import numpy as np
from collections import Counter, OrderedDict
import tqdm

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import FeatureSeqEmbLayer
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss

from torch.nn.init import xavier_uniform_, xavier_normal_

from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader


class CustomFeatureSeqEmbLayer(FeatureSeqEmbLayer):

    def forward(self, user_idx, item_idx):
        sparse_embedding, dense_embedding = self.embed_input_fields(user_idx, item_idx)
        sparse_embedding = sparse_embedding['item']
        dense_embedding = dense_embedding['item']
        # concat the sparse embedding and float embedding
        feature_table = []
        if sparse_embedding is not None:
            feature_table.append(sparse_embedding)
        if dense_embedding is not None:
            feature_table.append(dense_embedding)

        feature_table = torch.cat(feature_table, dim=-2)
        # [batch len num_features hidden_size]
        table_shape = feature_table.shape

        feat_num, embedding_size = table_shape[-2], table_shape[-1]
        feature_emb = feature_table.view(table_shape[:-2] + (feat_num * embedding_size,))
        return feature_emb


class GCN(torch.nn.Module):

    def __init__(self, input_size, hidden_size, embedding_size, num_layers=2, heads=1):
        super().__init__()

        assert num_layers >= 1
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(input_size, hidden_size,
                                    heads, edge_dim=1))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(heads * hidden_size, hidden_size, heads, edge_dim=1))  # edge_dim = 1 for edge weights
        self.convs.append(
            GATv2Conv(heads * hidden_size, embedding_size, heads=1, edge_dim=1,
                      concat=False))

        self.skip_lins = nn.ModuleList(
            [nn.Linear(input_size, heads * hidden_size)] * (num_layers - 2) + [nn.Linear(input_size, embedding_size)])
        self.activations = nn.ModuleList(
            [nn.PReLU(heads * hidden_size)] * (num_layers - 1) + [nn.PReLU(embedding_size)])

    def forward(self, x, edge_index, edge_weight):
        h = x
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index, edge_weight)
            if i != 0:
                h = h + self.skip_lins[i - 1](x)
            h = self.activations[i](h)
        return h

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for skip in self.skip_lins:
            skip.reset_parameters()
        for m in self.activations:
            m.weight.data.fill_(0.25)


class SRGCN(SequentialRecommender):
    def __init__(self, config, dataset):
        super(SRGCN, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.n_gcn_layers = config['n_gcn_layers']
        self.n_gcn_heads = config['n_gcn_heads']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        self.n_user = dataset.num(self.USER_ID)
        self.embed_user = config['embed_user']

        self.n_neighbors = config['n_neighbors']
        self.pooling_mode = config['pooling_mode']
        self.selected_features = config['selected_features']
        self.device = config['device']
        self.num_feature_field = len(config['selected_features'])
        self.items = dataset.item_feat.interaction['item_id']

        self.feature_embed_layer = CustomFeatureSeqEmbLayer(
            dataset, self.embedding_size, self.selected_features, self.pooling_mode, self.device
        )

        self.item_graph = self._init_graph()
        self.gcn = GCN(input_size=(self.num_feature_field * self.embedding_size) + 1,  # add node degree feature
                       hidden_size=self.hidden_size,
                       embedding_size=self.embedding_size,
                       num_layers=self.n_gcn_layers,
                       heads=self.n_gcn_heads)

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        # TODO self.session_embedding
        self.user_embedding = nn.Embedding(self.n_user, self.embedding_size)

        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.embed_user:
            self.dense = nn.Linear(self.hidden_size * 2, self.embedding_size)
        else:
            self.dense = nn.Linear(self.hidden_size, self.embedding_size)
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _init_graph(self):
        node_features = self.feature_embed_layer(None, self.items)
        node_features = torch.cat((node_features, torch.ones(node_features.shape[0], 1).to(self.device)),
                                  dim=1)  # add placeholder node degree feature

        edge_weights = torch.ones(len(self.items))

        edge_index = torch.stack((self.items, self.items)).long()

        data = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weights)
        return data

    def _update_graph(self, item_seq):
        edges_src = self.item_graph.edge_index[0].tolist()
        edges_dst = self.item_graph.edge_index[1].tolist()
        items = item_seq.flatten().unique()[:-1]
        items = items[items != 0]  # ignore 0
        for item in items:
            list_mask = torch.sum(torch.where(item_seq == item, item_seq, 0), dim=1).bool()
            for item_list in item_seq[list_mask]:
                session_items = item_list[item_list != 0]
                edges_dst.extend(session_items.tolist())
                edges_src.extend([item.item()] * len(session_items))

        # unique edges and edge weights
        edges_tpl = list(zip(edges_src, edges_dst))
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
        edge_weights_norm = np.array(
            [weight / node_degrees[edges_src_lst[i]] for i, weight in enumerate(edge_weights)])

        node_features = self.feature_embed_layer(None, self.items)
        node_features = torch.cat((node_features, torch.Tensor(node_degrees_norm).unsqueeze(dim=1).to(self.device)),
                                  dim=1)

        edge_weights = torch.from_numpy(np.concatenate((edge_weights_norm, edge_weights_norm))).type(torch.FloatTensor)
        edges_src = torch.from_numpy(np.array(list(edges_src_lst) + list(edges_dst_lst)))
        edges_dst = torch.from_numpy(np.array(list(edges_dst_lst) + list(edges_src_lst)))

        edge_index = torch.stack((edges_src, edges_dst)).long()

        data = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weights)
        return data

    def _update_node_embeddings(self, item_seq):
        input_nodes = item_seq.flatten().unique()
        train_loader = NeighborLoader(self.item_graph, num_neighbors=self.n_neighbors, shuffle=True,
                                      input_nodes=input_nodes,
                                      batch_size=item_seq.shape[0])
        data_loader = NeighborLoader(self.item_graph, num_neighbors=[0], shuffle=False, batch_size=item_seq.shape[0])
        self.gcn.train()
        for batch in train_loader:
            batch = batch.to(self.device)
            _ = self.gcn(batch.x, batch.edge_index, batch.edge_weight)

        self.gcn.eval()
        reps = []
        for batch in data_loader:
            batch = batch.to(self.device)
            reps.append(self.gcn(batch.x, batch.edge_index, batch.edge_weight))
        emb_weights = torch.cat(reps, dim=0)
        return emb_weights

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq, item_seq_len, user):
        self.item_graph = self._update_graph(item_seq)
        gcn_emb_weights = self._update_node_embeddings(item_seq)
        self.item_embedding.weight = torch.nn.Parameter(gcn_emb_weights)

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_seq_emb = self.item_embedding(item_seq)
        input_emb = item_seq_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        trm_output = trm_output[-1]

        if self.embed_user:
            user_emb = self.user_embedding.weight[user].clone() * torch.prod(item_seq_emb, 1)
            self.user_embedding.weight[user].data = user_emb
            user_emb = self.user_embedding(user)
            user_emb_dropout = self.emb_dropout(user_emb)
            user_seq_emb_dropout = user_emb_dropout.unsqueeze(1).repeat(1, trm_output.shape[1], 1)
            dense_output = self.dense(torch.cat((trm_output, user_seq_emb_dropout), -1))
        else:
            dense_output = self.dense(trm_output)

        output = self.gather_indexes(dense_output, item_seq_len - 1)
        return output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user = interaction[self.USER_ID]
        seq_output = self.forward(item_seq, item_seq_len, user)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)  # [B H]
            neg_items_emb = self.item_embedding(neg_items)  # [B H]
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user = interaction[self.USER_ID]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len, user)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user = interaction[self.USER_ID]
        seq_output = self.forward(item_seq, item_seq_len, user)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
