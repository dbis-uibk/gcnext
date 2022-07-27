from gnn_dataset.dataset import BaseSessionGNNDatasetPyG

inter_names = ['session_id:token', 'item_id:token', 'timestamp:float', 'item_id_list:token_seq', 'session_id_original:token']
item_names = ['item_id:token', 'duration:float', 'release_year:token', 'us_popularity_estimate:float',
              'acousticness:float', 'beat_strength:float', 'bounciness:float', 'danceability:float', 'dyn_range_mean:float',
              'flatness:float', 'instrumentalness:float', 'key:token', 'liveness:float', 'loudness:float', 'mechanism:float',
              'mode:token', 'organism:float', 'speechiness:float', 'tempo:float', 'time_signature:token', 'valence:float',
              'acoustic_vector:float_seq']
item_num_feature_names = ['duration:float', 'us_popularity_estimate:float',
              'acousticness:float', 'beat_strength:float', 'bounciness:float', 'danceability:float', 'dyn_range_mean:float',
              'flatness:float', 'instrumentalness:float', 'liveness:float', 'loudness:float', 'mechanism:float',
              'organism:float', 'speechiness:float', 'tempo:float', 'valence:float']
item_cat_feature_names = ['release_year:token', 'key:token', 'mode:token', 'time_signature:token']
item_seq_feature_name = None
item_emb_feature_names = None#['acoustic_vector:float_seq']

class MSSDDatasetPyG(BaseSessionGNNDatasetPyG):
    def __init__(self, dataset_name='mssd', pre_dataset_path=None, save_dir=None, transform=None, pre_transform=None,
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
    dataset = MSSDDatasetPyG(pre_dataset_path='../../data/mssd/', use_chached=False)
    graph = dataset[0]

    print(graph)
