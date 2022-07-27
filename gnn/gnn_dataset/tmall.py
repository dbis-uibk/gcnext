from gnn_dataset.dataset import BaseSessionGNNDatasetPyG

inter_names = ['session_id:token', 'item_id:token', 'timestamp:float', 'item_id_list:token_seq', 'session_id_original:token']
item_names = ['item_id:token', 'category_id:token', 'seller_id:token', 'brand_id:token']
item_num_feature_names = None
item_cat_feature_names = ['category_id:token', 'seller_id:token', 'brand_id:token']
item_seq_feature_name = None
item_emb_feature_names = None


class TmallDatasetPyG(BaseSessionGNNDatasetPyG):
    def __init__(self, dataset_name='tmall', pre_dataset_path=None, save_dir=None, transform=None, pre_transform=None,
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
    dataset = TmallDatasetPyG(pre_dataset_path='../../data/tmall/', save_dir="../../data/tmall-pyg/", use_chached=False)
    print(dataset)