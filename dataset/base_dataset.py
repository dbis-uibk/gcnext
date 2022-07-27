"""
Reference: https://github.com/RUCAIBox/RecSysDatasets/blob/master/conversion_tools/src/base_dataset.py
"""

import os
import random

import numpy as np
import pandas as pd

from tqdm import tqdm

pd.options.mode.chained_assignment = None


class BaseDataset(object):
    def __init__(self, input_path, output_path, dataset_name):
        super(BaseDataset, self).__init__()

        self.dataset_name = dataset_name
        self.input_path = input_path
        self.output_path = output_path
        self.check_output_path()

        # output file
        self.output_inter_file, self.output_inter_file_train, self.output_inter_file_valid, self.output_inter_file_test = self.get_inter_output_files()
        self.output_item_file, self.output_item_file_train, self.output_item_file_valid, self.output_item_file_test = self.get_item_output_files()

        # selected feature fields
        self.inter_fields = {}
        self.item_fields = {}

        # df feature names
        self.df_inter_names_full = ['session_id', 'item_id', 'timestamp']
        self.df_inter_names = []
        self.df_item_names = []

        self.full = None
        self.train = None
        self.valid = None
        self.test = None

        self.item_full = None

        self.file_names = ['train', 'valid', 'test']

        self.min_item_support = None
        self.min_session_length = None
        self.max_session_length = None
        self.valid_size = None
        self.test_size = None

    def check_output_path(self):
        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)

    def get_inter_output_files(self):
        output_inter_file = os.path.join(self.output_path, self.dataset_name + '.inter')
        output_inter_file_train = os.path.join(self.output_path, self.dataset_name + '.train.inter')
        output_inter_file_valid = os.path.join(self.output_path, self.dataset_name + '.valid.inter')
        output_inter_file_test = os.path.join(self.output_path, self.dataset_name + '.test.inter')
        return output_inter_file, output_inter_file_train, output_inter_file_valid, output_inter_file_test

    def get_item_output_files(self):
        output_item_file = os.path.join(self.output_path, self.dataset_name + '.item')
        output_item_file_train = os.path.join(self.output_path, self.dataset_name + '.train.item')
        output_item_file_valid = os.path.join(self.output_path, self.dataset_name + '.valid.item')
        output_item_file_test = os.path.join(self.output_path, self.dataset_name + '.test.item')
        return output_item_file, output_item_file_train, output_item_file_valid, output_item_file_test

    def convert_inter(self):
        fout = open(self.output_inter_file, 'w', encoding='utf-8')
        fout.write('\t'.join([self.inter_fields[i] for i in range(len(self.df_inter_names_full))]) + '\n')
        fout.close()

        self.full[self.df_inter_names_full].to_csv(self.output_inter_file, mode='a',
                                                        header=False, encoding='utf-8', sep='\t',
                                                        index=False)

        for name in self.file_names:
            # convert item id list to string
            getattr(self, name)['item_id_list'] = getattr(self, name)['item_id_list'].apply(lambda x: ' '.join(map(str, list(x))))
            fout = open(getattr(self, "output_inter_file_" + name), 'w', encoding='utf-8')
            fout.write('\t'.join([self.inter_fields[i] for i in range(len(self.inter_fields))]) + '\n')
            fout.close()

            getattr(self, name)[self.df_inter_names].to_csv(getattr(self, "output_inter_file_" + name), mode='a',
                                                                              header=False, encoding='utf-8', sep='\t',
                                                                              index=False)
        print('.inter is done!')

    def convert_item(self):
        fout = open(self.output_item_file, 'w', encoding='utf-8')
        fout.write('\t'.join([self.item_fields[i] for i in range(len(self.item_fields))]) + '\n')
        fout.close()

        self.item_full[self.df_item_names].to_csv(self.output_item_file, mode='a',
                                                        header=False, encoding='utf-8', sep='\t',
                                                        index=False)
        for name in self.file_names:
            fout = open(getattr(self, "output_item_file_" + name), 'w', encoding='utf-8')
            fout.write('\t'.join([self.item_fields[i] for i in range(len(self.item_fields))]) + '\n')
            fout.close()

            getattr(self, 'item_' + name)[self.df_item_names].to_csv(getattr(self, "output_item_file_" + name), mode='a',
                                                            header=False, encoding='utf-8', sep='\t',
                                                            index=False)
        print('.item is done!')

    def prepare_dataset(self, df_inter, size=None):
        df_inter = df_inter.dropna(subset=['timestamp', 'item_id', 'session_id'])
        df_inter = df_inter.drop_duplicates()

        # drop item interactions below item support
        df_inter['count_item_support'] = df_inter.groupby(['item_id'])['item_id'].transform('count')
        df_inter = df_inter.drop(df_inter[df_inter.count_item_support < self.min_item_support].index)

        # drop sessions not having min_session_length
        df_inter['count_session_items'] = df_inter.groupby(['session_id'])['session_id'].transform('count')
        df_inter = df_inter.drop(df_inter[df_inter.count_session_items < self.min_session_length].index)

        # slice to max_session_length
        df_inter = df_inter.sort_values(by=['session_id', 'timestamp'])
        df_inter['cumcount_session_items'] = df_inter.groupby(['session_id'])['session_id'].transform('cumcount')
        df_inter = df_inter.drop(df_inter[df_inter.cumcount_session_items >= self.max_session_length].index)

        if size:
            df_inter = df_inter.sort_values(by=['session_id', 'timestamp'], ascending=False)
            df_inter = df_inter.head(int(len(df_inter) * size))

        print("factorize items...")
        df_inter['item_id'], _ = df_inter['item_id'].factorize()
        df_inter['item_id'] = df_inter['item_id'] + 1  # item ids should start from 1 (RecBole)

        df_inter[['timestamp', 'item_id', 'session_id']] = df_inter[['timestamp', 'item_id', 'session_id']].astype('Int64')

        # create item df
        self.item_full = df_inter[self.df_item_names].drop_duplicates(subset='item_id', keep="first").copy()

        df_inter['session_id'], _ = df_inter['session_id'].factorize()
        df_inter['session_id'] = df_inter['session_id'] + 1
        print("# sessions: ", df_inter['session_id'].max())

        self.full = df_inter

        # create item list ids of sessions
        print("data sequencing and augmentation...")
        df_inter = self.data_sequencing(df_inter, augmentation=True, reindex=True)

        # split inter dataset
        self.train, self.valid, self.test = self.split_df(df=df_inter, valid_size=self.valid_size, test_size=self.test_size)

        # delete augmentations from valid & test
        self.valid = self.valid[self.valid['last_session_entry'] == True]
        self.test = self.test[self.test['last_session_entry'] == True]

        # split item dataset
        train_items = np.concatenate(self.train['item_id_list'].values.tolist() + [self.train['item_id'].tolist()])
        valid_items = np.concatenate(self.valid['item_id_list'].values.tolist() + [self.valid['item_id'].tolist()])
        test_items = np.concatenate(self.test['item_id_list'].values.tolist() + [self.test['item_id'].tolist()])
        self.item_train = self.item_full[self.item_full['item_id'].isin(train_items)]
        self.item_valid = self.item_full[self.item_full['item_id'].isin(valid_items)]
        self.item_test = self.item_full[self.item_full['item_id'].isin(test_items)]
        self.item_full = self.item_full[
            self.item_full['item_id'].isin(np.concatenate((train_items, valid_items, test_items)))]

        print(self.full.shape, self.train.shape, self.valid.shape, self.test.shape)

    def split_df(self, df, valid_size=0.1, test_size=0.1):

        df = df.sort_values(by=['timestamp', 'session_id'], ascending=False)

        test, valid, train = np.split(df, [int(test_size * len(df)),
                                                 int(test_size * len(df)) + int(valid_size * len(df))])

        train = train.sort_values(by=['session_id'], ascending=True)
        valid = valid.sort_values(by=['session_id'], ascending=True)
        test = test.sort_values(by=['session_id'], ascending=True)

        # drop sessions from valid or test if in train or valid
        valid = valid[~valid.session_id.isin(train.session_id)]
        test = test[~test.session_id.isin(train.session_id)]
        test = test[~test.session_id.isin(valid.session_id)]

        return train, valid, test

    def data_sequencing(self, df, augmentation=True, reindex=True):
        """
        Augmentation processing for sequential dataset.

        E.g., ``u1`` has purchase sequence ``<i1, i2, i3, i4>``,
        then after augmentation, we will generate three cases.

        ``u1, <i1> | i2``

        (Which means given user_id ``u1`` and item_seq ``<i1>``,
        we need to predict the next item ``i2``.)

        The other cases are below:

        ``u1, <i1, i2> | i3``

        ``u1, <i1, i2, i3> | i4``
        """
        df = df.sort_values(by=['session_id', 'timestamp'], ascending=True)

        df_sessions_items = df.groupby(['session_id'], as_index=False).agg({'item_id': lambda x: list(x)})
        df_sessions_items = df_sessions_items.rename(columns={"item_id": "item_id_list"})
        df = pd.merge(df, df_sessions_items, on='session_id', how='outer')

        df['last_session_entry'] = False

        curr_session_id = -1
        item_counter = 0
        item_list_ids = []
        for idx, inter in tqdm(df[['session_id', 'item_id_list']].iterrows(), total=df.shape[0]):
            if inter['session_id'] == curr_session_id:
                item_counter += 1
            else:
                if idx != 0:
                    df['last_session_entry'][idx-1] = True
                item_counter = 0
                curr_session_id = inter['session_id']
            item_list_ids.append(inter['item_id_list'][:item_counter])

        df['item_id_list'] = item_list_ids

        # drop first item in session from interactions
        df = df[df['item_id_list'].map(len) > 0]

        # delete all produced inter-sequences
        if not augmentation:
            df = df[df['last_session_entry'] == True]

        # factorize all augmented interactions to different session ids
        if reindex:
            df['session_id_original'] = df['session_id']
            df['session_id'] = df.reset_index(level=0).index

        return df