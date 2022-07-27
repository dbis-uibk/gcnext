from base_dataset import BaseDataset
import argparse
import pandas as pd
import numpy as np
import os
import glob

class MSSDDataset(BaseDataset):
    def __init__(self, input_path, output_path, min_item_support=5, min_session_length=2, max_session_length=50,
                 valid_size=0.1, test_size=0.1):
        super(MSSDDataset, self).__init__(input_path, output_path, dataset_name='mssd')
        self.dataset_name = 'mssd'

        self.min_item_support = min_item_support
        self.min_session_length = min_session_length
        self.max_session_length = max_session_length
        self.valid_size = valid_size
        self.test_size = test_size

        self.sep = ','
        self.inter_fields = {0: 'session_id:token',
                             1: 'item_id:token',
                             2: 'timestamp:float',
                             3: 'item_id_list:token_seq',
                             4: 'session_id_original:token',
                             }

        self.item_fields = {0: 'item_id:token',
                            1: 'duration:float',
                            2: 'release_year:token',
                            3: 'us_popularity_estimate:float',
                            4: 'acousticness:float',
                            5: 'beat_strength:float',
                            6: 'bounciness:float',
                            7: 'danceability:float',
                            8: 'dyn_range_mean:float',
                            9: 'flatness:float',
                            10: 'instrumentalness:float',
                            11: 'key:token',
                            12: 'liveness:float',
                            13: 'loudness:float',
                            14: 'mechanism:float',
                            15: 'mode:token',
                            16: 'organism:float',
                            17: 'speechiness:float',
                            18: 'tempo:float',
                            19: 'time_signature:token',
                            20: 'valence:float',
                            21: 'acoustic_vector:float_seq',
                            }

        self.inter_path = os.path.join(self.input_path, 'training_set')
        self.item_path = os.path.join(self.input_path, "track_features")

        # df feature names
        self.df_inter_names = ['session_id', 'item_id', 'timestamp', 'item_id_list', 'session_id_original']
        self.df_item_names = ['item_id', 'duration', 'release_year', 'us_popularity_estimate', 'acousticness', 'beat_strength',
                            'bounciness', 'danceability', 'dyn_range_mean', 'flatness', 'instrumentalness', 'key', 'liveness',
                            'loudness', 'mechanism', 'mode', 'organism', 'speechiness', 'tempo', 'time_signature', 'valence',
                              'acoustic_vector']

        print("read files...")
        df_item = pd.concat(map(pd.read_csv, glob.glob(os.path.join(self.item_path, "*.csv"))))
        df_inter = pd.concat(map(pd.read_csv, glob.glob(os.path.join(self.inter_path, "*.csv"))))
        #df_inter = pd.read_csv(os.path.join(self.inter_path, "log_0_20180715_000000000000.csv"))

        df_inter = df_inter[['session_id', 'session_position', 'track_id_clean', 'date', 'hour_of_day']]
        df_inter = df_inter.dropna()

        print("assign acoustic vector...")
        df_item = df_item.assign(acoustic_vector=df_item.acoustic_vector_0.astype(str) + ' ' +
                                                   df_item.acoustic_vector_1.astype(str) + ' ' +
                                                   df_item.acoustic_vector_2.astype(str) + ' ' +
                                                   df_item.acoustic_vector_3.astype(str) + ' ' +
                                                   df_item.acoustic_vector_4.astype(str) + ' ' +
                                                   df_item.acoustic_vector_5.astype(str) + ' ' +
                                                   df_item.acoustic_vector_6.astype(str) + ' ' +
                                                   df_item.acoustic_vector_7.astype(str))

        # use only one week
        df_inter['date'] = pd.to_datetime(df_inter['date'])
        df_inter = df_inter[(df_inter['date'] >= pd.Timestamp('2018-07-15')) &
                            (df_inter['date'] <= pd.Timestamp('2018-07-17'))]

        df_inter = pd.merge(df_inter, df_item,  left_on='track_id_clean', right_on='track_id', how='left')

        df_inter = df_inter.rename(columns={"track_id_clean": "item_id"})
        df_inter.hour_of_day = df_inter.hour_of_day.fillna(0).astype(int)

        print("factorize sessions...")
        df_inter['session_id'], _ = df_inter['session_id'].factorize()
        df_inter['session_id'] = df_inter['session_id'] + 1

        # create timestamp from date, hour of day, session position
        print("create timestamps...")
        df_inter['timestamp'] = pd.to_datetime(df_inter['date'].astype(str) + ' ' + df_inter['hour_of_day'].astype(str)
                                               + ':' + (df_inter['session_position']).astype(str)).values.astype(np.int64) // 10 ** 9

        self.prepare_dataset(df_inter, size=0.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mssd')
    parser.add_argument('--input_path', type=str, default="../data/mssd")
    parser.add_argument('--output_path', type=str, default="../data/mssd")

    parser.add_argument('--convert_inter', action='store_true', default=True)
    parser.add_argument('--convert_item', action='store_true', default=True)

    args = parser.parse_args()

    assert args.input_path is not None, 'input_path can not be None, please specify the input_path'
    assert args.output_path is not None, 'output_path can not be None, please specify the output_path'

    input_args = [args.input_path, args.output_path]
    datasets = MSSDDataset(*input_args)

    if args.convert_inter:
        datasets.convert_inter()
    if args.convert_item:
        datasets.convert_item()
