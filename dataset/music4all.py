from base_dataset import BaseDataset
import argparse
import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import LabelEncoder


def flatten_list(t):
    return [item for sublist in t for item in sublist]


class Music4AllDataset(BaseDataset):
    def __init__(self, input_path, output_path, min_item_support=5, min_session_length=2, max_session_length=50,
                 valid_size=0.1, test_size=0.1, mul_feat_vecs=False):
        super().__init__(input_path, output_path, dataset_name='music4all')

        self.min_item_support = min_item_support
        self.min_session_length = min_session_length
        self.max_session_length = max_session_length
        self.valid_size = valid_size
        self.test_size = test_size

        self.input_metadata = os.path.join(self.input_path, "id_metadata.csv")
        self.input_lang = os.path.join(self.input_path, "id_lang.csv")
        self.input_tags = os.path.join(self.input_path, "id_tags.csv")
        self.input_info = os.path.join(self.input_path, "id_information.csv")
        self.input_interactions = os.path.join(self.input_path, "listening_history.csv")

        self.input_chroma_bow = os.path.join(self.input_path, "multimodal", "id_chroma_bow.csv")
        self.input_emobase_bow = os.path.join(self.input_path, "multimodal", "id_emobase_bow.csv")
        self.input_ivec = os.path.join(self.input_path, "multimodal", "id_ivec256.csv")
        self.input_lyrics_sentiment = os.path.join(self.input_path, "multimodal", "id_lyrics_sentiment_functionals.csv")
        self.input_mean_cov = os.path.join(self.input_path, "multimodal", "id_mean_cov.csv")
        self.input_mfcc_bow = os.path.join(self.input_path, "multimodal", "id_mfcc_bow.csv")
        self.input_vad_cov = os.path.join(self.input_path, "multimodal", "id_vad_bow.csv")

        self.sep = '\t'

        # selected feature fields
        self.inter_fields = {0: 'session_id:token',
                             1: 'item_id:token',
                             2: 'timestamp:float',
                             3: 'user_id:token',
                             4: 'item_id_list:token_seq',
                             5: 'session_id_original:token',
                             }

        self.item_fields = {0: 'item_id:token',
                            1: 'popularity:float',
                            2: 'release:token',
                            3: 'danceability:float',
                            4: 'energy:float',
                            5: 'key:token',
                            6: 'mode:token',
                            7: 'valence:float',
                            8: 'tempo:float',
                            9: 'duration_ms:float',
                            10: 'language:token',
                            11: 'artist:token',
                            12: 'album_name:token',
                            13: 'tags:token_seq',
                            14: 'chroma:float_seq',
                            15: 'emobase:float_seq',
                            16: 'ivec:float_seq',
                            17: 'lyrics:float_seq',
                            18: 'mean:float_seq',
                            19: 'mfcc:float_seq',
                            20: 'vad:float_seq',
                            }

        # df feature names
        self.df_inter_names = ['user_id', 'item_id', 'timestamp', 'session_id', 'item_id_list', 'session_id_original']
        if mul_feat_vecs:
            self.df_item_names = ['item_id', 'popularity', 'release', 'danceability', 'energy', 'key', 'mode',
                                  'valence', 'tempo', 'duration_ms', 'lang', 'artist', 'album_name', 'tags', 'feat_vec']
        else:
            self.df_item_names = ['item_id', 'popularity', 'release', 'danceability', 'energy', 'key', 'mode', 'valence',
                                  'tempo', 'duration_ms', 'lang', 'artist', 'album_name', 'tags', 'chroma', 'emobase', 'ivec',
                                  'lyrics', 'mean', 'mfcc', 'vad']

        print("read item files...")
        df_item = pd.read_csv(self.input_metadata, sep=self.sep)
        df_lang = pd.read_csv(self.input_lang, sep=self.sep)
        df_tags = pd.read_csv(self.input_tags, sep=self.sep)
        df_info = pd.read_csv(self.input_info, sep=self.sep)

        df_chroma = pd.read_csv(self.input_chroma_bow, sep=",")
        df_emobase = pd.read_csv(self.input_emobase_bow, sep=",")
        df_ivec = pd.read_csv(self.input_ivec, sep=",")
        df_lyrics = pd.read_csv(self.input_lyrics_sentiment, sep=",")
        df_mean = pd.read_csv(self.input_mean_cov, sep=",")
        df_mfcc = pd.read_csv(self.input_mfcc_bow, sep=",")
        df_vad = pd.read_csv(self.input_vad_cov, sep=",")

        if mul_feat_vecs:
            lst_df_multimodal = [df_chroma, df_emobase, df_mfcc, df_vad]
            vec = lst_df_multimodal[0]
            vec = vec.set_index('ID')
            for i, mdf in enumerate(lst_df_multimodal[1:]):
                mdf = mdf.set_index('ID')
                vec = vec * mdf.values
            vec['feat_vec'] = vec.values.tolist()
            vec['feat_vec'] = vec['feat_vec'].apply(lambda x: ' '.join(['{:.8f}'.format(emb) for emb in x]))
            vec = vec.reset_index()
            df_item = df_item.merge(vec[['ID', 'feat_vec']], left_on='id', right_on='ID')
        else:
            lst_df_multimodal = [df_chroma, df_emobase, df_ivec, df_lyrics, df_mean, df_mfcc, df_vad]
            emb_names = ['chroma', 'emobase', 'ivec', 'lyrics', 'mean', 'mfcc', 'vad']

            for i, mdf in enumerate(lst_df_multimodal):
                c_name = emb_names[i]
                mdf = mdf.rename(columns={"ID": "id"})
                mdf[c_name] = mdf.loc[:, mdf.columns != 'id'].values.tolist()
                mdf[c_name] = mdf[c_name].apply(lambda x: ' '.join(['{:.8f}'.format(emb) for emb in x]))
                df_item = df_item.merge(mdf[['id', c_name]], on='id')

        df_tags['tags'] = df_tags['tags'].apply(lambda x: ' '.join(x.split(",")))
        df_item = df_item.merge(df_lang, on='id').merge(df_tags, on='id').merge(df_info, on='id')

        print("preprocessing...")
        df_item['lang'], _ = df_item['lang'].factorize()
        df_item['artist'], _ = df_item['artist'].factorize()
        df_item['album_name'], _ = df_item['album_name'].factorize()

        df_inter = pd.read_csv(self.input_interactions, sep=self.sep)
        df_inter = pd.merge(df_inter, df_item, left_on='song', right_on='id')
        df_inter = df_inter.rename(columns={"user": "user_id", "id": "item_id"})

        # create sessions
        print("create sessions...")
        df_inter['timestamp'] = pd.to_datetime(df_inter['timestamp'])
        df_inter.sort_values(by=['user_id', 'timestamp'], inplace=True)
        cond1 = df_inter.timestamp - df_inter.timestamp.shift(1) > pd.Timedelta(30, 'm')
        cond2 = df_inter.user_id != df_inter.user_id.shift(1)
        df_inter['session_id'] = (cond1 | cond2).cumsum()

        df_inter['timestamp'] = df_inter['timestamp'].values.astype(np.int64) // 10 ** 9

        self.prepare_dataset(df_inter, size=0.01)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='music4all')
    parser.add_argument('--input_path', type=str, default="../data/music4all")
    parser.add_argument('--output_path', type=str, default="../data/music4all")

    parser.add_argument('--convert_inter', action='store_true', default=True)
    parser.add_argument('--convert_item', action='store_true', default=True)

    args = parser.parse_args()

    assert args.input_path is not None, 'input_path can not be None, please specify the input_path'
    assert args.output_path is not None, 'output_path can not be None, please specify the output_path'

    input_args = [args.input_path, args.output_path]
    datasets = Music4AllDataset(*input_args)

    if args.convert_inter:
        datasets.convert_inter()
    if args.convert_item:
        datasets.convert_item()