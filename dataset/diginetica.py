from base_dataset import BaseDataset
import argparse
import pandas as pd
import numpy as np
import os


class DigineticaDataset(BaseDataset):
    def __init__(self, input_path, output_path, min_item_support=5, min_session_length=2, max_session_length=50,
                 valid_size=0.1, test_size=0.1):
        super().__init__(input_path, output_path, dataset_name='diginetica')

        self.min_item_support = min_item_support
        self.min_session_length = min_session_length
        self.max_session_length = max_session_length
        self.valid_size = valid_size
        self.test_size = test_size

        self.input_file_prod = os.path.join(self.input_path, "products.csv")
        self.input_file_prod_cate = os.path.join(self.input_path, "product-categories.csv")
        self.input_file_item_views = os.path.join(self.input_path, "train-item-views.csv")

        self.sep = ';'
        self.categories_names = ['item_id', 'category_id']
        self.product_names = ['item_id', 'pricelog2', 'name_tokens']
        self.view_names = ['session_id', 'user_id', 'item_id', 'timeframe', 'eventdate']

        # selected feature fields
        self.inter_fields = {0: 'session_id:token',
                             1: 'item_id:token',
                             2: 'timestamp:float',
                             3: 'item_id_list:token_seq',
                             4: 'session_id_original:token',
                             }

        self.item_fields = {0: 'item_id:token',
                            1: 'category_id:token',
                            2: 'name_tokens:token_seq',
                            3: 'pricelog2:float'
                            }

        # df feature names
        self.df_inter_names = ['session_id', 'item_id', 'timestamp', 'item_id_list', 'session_id_original']
        self.df_item_names = ['item_id', 'category_id', 'name_tokens', 'pricelog2']

        print("read file...")
        df_item = pd.read_csv(self.input_file_prod, sep=self.sep, names=self.product_names, header=0)
        df_cat = pd.read_csv(self.input_file_prod_cate, sep=self.sep, names=self.categories_names, header=0)
        df_item = pd.merge(df_item, df_cat, on='item_id', how='outer')

        df_inter = pd.read_csv(self.input_file_item_views, sep=self.sep, names=self.view_names, header=0)
        df_inter = pd.merge(df_item, df_inter, on='item_id', how='outer')

        df_inter['timestamp'] = pd.to_datetime(df_inter['eventdate']).values.astype(np.int64) // 10 ** 9
        df_inter['timestamp'] = df_inter['timestamp'].add(df_inter['timeframe'])

        # convert name tokens to sep string
        df_inter['name_tokens'] = df_inter['name_tokens'].apply(lambda x: ' '.join(x.split(",")))

        self.prepare_dataset(df_inter, size=0.1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='diginetica')
    parser.add_argument('--input_path', type=str, default="../data/diginetica")
    parser.add_argument('--output_path', type=str, default="../data/diginetica")

    parser.add_argument('--convert_inter', action='store_true', default=True)
    parser.add_argument('--convert_item', action='store_true', default=True)

    args = parser.parse_args()

    assert args.input_path is not None, 'input_path can not be None, please specify the input_path'
    assert args.output_path is not None, 'output_path can not be None, please specify the output_path'

    input_args = [args.input_path, args.output_path]
    datasets = DigineticaDataset(*input_args)

    if args.convert_inter:
        datasets.convert_inter()
    if args.convert_item:
        datasets.convert_item()
