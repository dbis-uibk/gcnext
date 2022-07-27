import torch
import numpy as np

from gnn_dataset.diginetica import DigineticaDatasetPyG
from gnn_dataset.yoochoose import YoochooseDatasetPyG
from gnn_dataset.globo import GLoboDatasetPyG
from gnn_dataset.mssd import MSSDDatasetPyG
from gnn_dataset.music4all import Music4AllDatasetPyG
from gnn_dataset.tmall import TmallDatasetPyG

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def get_pyg_dataset(name, save_dir=None, use_cached=True, inductive=True):
    dataset = None
    if name == 'yoochoose-4':
        dataset = YoochooseDatasetPyG(dataset_name='yoochoose-4', pre_dataset_path='../data/yoochoose-4/', save_dir=save_dir,
                                      use_chached=use_cached, inductive=inductive)
    elif name == 'yoochoose-64':
        dataset = YoochooseDatasetPyG(dataset_name='yoochoose-64', pre_dataset_path='../data/yoochoose-64/', save_dir=save_dir,
                                      use_chached=use_cached, inductive=inductive)
    elif name == 'diginetica':
        dataset = DigineticaDatasetPyG(dataset_name='diginetica', pre_dataset_path='../data/diginetica/', save_dir=save_dir,
                                       use_chached=use_cached, inductive=inductive)
    elif name == 'globo':
        dataset = GLoboDatasetPyG(dataset_name='globo', pre_dataset_path='../data/globo/', save_dir=save_dir,
                                  use_chached=use_cached, inductive=inductive)
    elif name == 'mssd':
        dataset = MSSDDatasetPyG(dataset_name='mssd', pre_dataset_path='../data/mssd/', save_dir=save_dir,
                                  use_chached=use_cached, inductive=inductive)
    elif name == 'music4all':
        dataset = Music4AllDatasetPyG(dataset_name='music4all', pre_dataset_path='../data/music4all/', save_dir=save_dir,
                                  use_chached=use_cached, inductive=inductive)
    elif name == 'tmall':
        dataset = TmallDatasetPyG(dataset_name='tmall', pre_dataset_path='../data/tmall/', save_dir=save_dir,
                                  use_chached=use_cached, inductive=inductive)

    return dataset


def output_embs(item_embs, dataset):
    output_emb_file = dataset.pre_dataset_path + f'{dataset.dataset_name}.itememb'
    output_train_emb_file = dataset.pre_dataset_path + f'{dataset.dataset_name}.train.itememb'
    output_valid_emb_file = dataset.pre_dataset_path + f'{dataset.dataset_name}.valid.itememb'
    output_test_emb_file = dataset.pre_dataset_path + f'{dataset.dataset_name}.test.itememb'

    df_item = pd.read_csv(dataset.pre_dataset_path + f'{dataset.dataset_name}.item', sep='\t', header=0,
                                names=dataset.item_names)
    df_train_item = pd.read_csv(dataset.pre_dataset_path + f'{dataset.dataset_name}.train.item', sep='\t', header=0,
                          names=dataset.item_names)
    df_valid_item = pd.read_csv(dataset.pre_dataset_path + f'{dataset.dataset_name}.valid.item', sep='\t', header=0,
                                names=dataset.item_names)
    df_test_item = pd.read_csv(dataset.pre_dataset_path + f'{dataset.dataset_name}.test.item', sep='\t', header=0,
                                names=dataset.item_names)

    item_encoder = LabelEncoder()
    item_encoder.classes_ = np.load(dataset.pre_dataset_path + f'{dataset.dataset_name}_item_labels.npy')
    item_ids = item_encoder.inverse_transform([i for i in range(len(item_embs))])
    emb_dict = dict()

    for i, id in enumerate(item_ids):
        emb_dict[id] = item_embs[i]

    write_embs(emb_dict, df_item, output_emb_file)
    write_embs(emb_dict, df_train_item, output_train_emb_file)
    write_embs(emb_dict, df_valid_item, output_valid_emb_file)
    write_embs(emb_dict, df_test_item, output_test_emb_file)

def write_embs(emb_dict, df_item, output_file):
    item_ids = df_item['item_id:token']
    emb_fields = {0: 'iid:token', 1: 'item_emb:float_seq'}

    fout = open(output_file, 'w', encoding='utf-8')
    fout.write('\t'.join([emb_fields[i] for i in range(len(emb_fields))]) + '\n')

    for id in item_ids:
        item_emb = emb_dict.get(id)
        fout.write('\t'.join([str(id), ' '.join(['{:.8f}'.format(emb) for emb in item_emb])]) + '\n')

    fout.close()
    #print(output_file + ' is done!')


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



