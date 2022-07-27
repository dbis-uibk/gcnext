from base_dataset import BaseDataset
import argparse
import pandas as pd
import numpy as np
import os

class YooChooseDataset(BaseDataset):
    def __init__(self, input_path, output_path, min_item_support=5, min_session_length=2, max_session_length=50,
                 valid_size=0.1, test_size=0.1):
        super(YooChooseDataset, self).__init__(input_path, output_path, dataset_name='yoochoose-64')
        self.dataset_name = 'yoochoose-64'

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
                            1: 'category_id:token'}

        self.inter_file = os.path.join(self.input_path, 'yoochoose-clicks.dat')

        self.inter_names = ['session_id', 'timestamp', 'item_id', 'category_id']

        # df feature names
        self.df_inter_names = ['session_id', 'item_id', 'timestamp', 'item_id_list', 'session_id_original']
        self.df_item_names = ['item_id', 'category_id']

        print("read file...")
        df_inter = pd.read_csv(self.inter_file, sep=self.sep, names=self.inter_names)

        df_inter['timestamp'] = pd.to_datetime(df_inter['timestamp']).values.astype(np.int64) // 10 ** 9

        self.prepare_dataset(df_inter, size=(1.5625/100)) # 1/64


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yoochoose-64')
    parser.add_argument('--input_path', type=str, default="../data/yoochoose-64")
    parser.add_argument('--output_path', type=str, default="../data/yoochoose-64")

    parser.add_argument('--convert_inter', action='store_true', default=True)
    parser.add_argument('--convert_item', action='store_true', default=True)

    args = parser.parse_args()

    assert args.input_path is not None, 'input_path can not be None, please specify the input_path'
    assert args.output_path is not None, 'output_path can not be None, please specify the output_path'

    input_args = [args.input_path, args.output_path]
    datasets = YooChooseDataset(*input_args)

    if args.convert_inter:
        datasets.convert_inter()
    if args.convert_item:
        datasets.convert_item()
