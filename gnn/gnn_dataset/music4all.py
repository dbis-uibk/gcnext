from gnn_dataset.dataset import BaseSessionGNNDatasetPyG

inter_names = ['session_id:token', 'item_id:token', 'timestamp:float', 'user_id:token', 'item_id_list:token_seq',
               'session_id_original:token']
item_names = ['item_id:token', 'popularity:float', 'release:token', 'danceability:float', 'energy:float', 'key:token',
            'mode:token', 'valence:float', 'tempo:float', 'duration_ms:float', 'language:token', 'artist:token',
            'album_name:token', 'tags:token_seq', 'chroma:float_seq', 'emobase:float_seq', 'ivec:float_seq',
              'lyrics:float_seq', 'mean:float_seq', 'mfcc:float_seq', 'vad:float_seq']


item_num_feature_names = ['popularity:float', 'danceability:float', 'energy:float', 'valence:float', 'tempo:float',
                          'duration_ms:float']
item_cat_feature_names = ['release:token', 'key:token', 'mode:token', 'language:token']#, 'artist:token']
                          #'album_name:token']
item_seq_feature_name = 'tags:token_seq'
item_emb_feature_names = ['ivec:float_seq'] #['chroma:float_seq', 'emobase:float_seq', 'ivec:float_seq', 'lyrics:float_seq',
                          #'mean:float_seq', 'mfcc:float_seq', 'vad:float_seq']

class Music4AllDatasetPyG(BaseSessionGNNDatasetPyG):
    def __init__(self, dataset_name='music4all', pre_dataset_path=None, save_dir=None, transform=None, pre_transform=None,
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
    dataset = Music4AllDatasetPyG(pre_dataset_path='../../data/music4all/', use_chached=False)
    graph = dataset[0]

    print(graph)
