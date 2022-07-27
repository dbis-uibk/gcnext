import argparse
import numpy as np
import csv   
from logging import getLogger
import os

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger

from models.utils import get_custom_model, get_config_from_file

parameter_dict = {
    'neg_sampling': None,
    'loss_type': 'CE',
    'MAX_ITEM_LIST_LENGTH': 50,
    'stopping_step': 4,
    'train_batch_size': 512,
    'eval_batch_size': 512,
    'metrics': ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision'],
    'topk': [10, 20],
    'valid_metric': 'MRR@20',
    'metric_decimal_place': 4,
    'seed': np.random.randint(low=42, high=2022),
    #'gpu_id': int(os.environ["CUDA_VISIBLE_DEVICES"])
}


def main(args):
    # configurations initialization
    config = get_config_from_file(args.config)
    config = Config(model=get_custom_model(config['model']), config_file_list=[args.config], config_dict=parameter_dict)

    # init random seed
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    # write config info into log
    logger.info(config)

    # dataset creating and filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = get_custom_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training and evaluation
    valid_score, valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=True
    )

    test_result = trainer.evaluate(test_data)
    print(test_result)

    # export run results
    fields = [config.model, dataset.dataset_name,
              test_result['mrr@10'], test_result['mrr@20'], 
              test_result['recall@10'], test_result['recall@20'],
              test_result['ndcg@10'], test_result['ndcg@20'], config.seed]

    with open('results.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Seq-Rec')
    parser.add_argument('--config', type=str, default='config/diginetica/gru4recgnn.yaml', help='config path')
    args = parser.parse_args()
    print(args)

    main(args=args)

