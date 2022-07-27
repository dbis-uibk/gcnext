from gnn_dataset.dataset import BaseSessionGNNDatasetPyG

inter_names = ['session_id:token', 'item_id:token', 'timestamp:float', 'item_id_list:token_seq', 'session_id_original:token']
item_names = ['item_id:token', 'category_id:token', 'name_tokens:token_seq', 'pricelog2:float']
item_num_feature_names = ['pricelog2:float']
item_cat_feature_names = ['category_id:token']
item_seq_feature_name = None #'name_tokens:token_seq' TODO make performant (currently one hot encoded...)
item_emb_feature_names = None


class DigineticaDatasetPyG(BaseSessionGNNDatasetPyG):
    def __init__(self, dataset_name='diginetica', pre_dataset_path=None, save_dir=None, transform=None, pre_transform=None,
                 second_lvl_session_edges=False, use_chached=True, inductive=True):
        super().__init__(dataset_name=dataset_name, pre_dataset_path=pre_dataset_path,
                         save_dir=save_dir, transform=transform, pre_transform=pre_transform,
                         inter_names=inter_names, item_names=item_names, item_num_feature_names=item_num_feature_names,
                         item_cat_feature_names=item_cat_feature_names, item_seq_feature_name=item_seq_feature_name,
                         item_emb_feature_names=item_emb_feature_names,
                         second_lvl_session_edges=second_lvl_session_edges, use_chached=use_chached,
                         inductive=inductive)

    def process(self):
        super().process()


if __name__ == '__main__':
    dataset = DigineticaDatasetPyG(pre_dataset_path='../../data/diginetica/', save_dir="../../data/diginetica-pyg/", use_chached=False)
    print(dataset)