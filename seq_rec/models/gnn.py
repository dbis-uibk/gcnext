import numpy as np

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss


class GNN(SequentialRecommender):

    def __init__(self, config, dataset):
        super(GNN, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']

        self.device = config['device']

        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)

        layers = list()
        layers.append(torch.nn.Dropout(p=self.dropout))
        input_dim = self.embedding_size
        for l in range(self.num_layers - 1):
            layers.append(torch.nn.Linear(input_dim, self.hidden_size))
            #layers.append(torch.nn.BatchNorm1d(self.hidden_size))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=self.dropout))
            input_dim = self.hidden_size
        layers.append(nn.Linear(self.hidden_size, self.embedding_size))
        self.mlp = torch.nn.Sequential(*layers)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(xavier_normal_initialization)

        pretrained_item_emb = dataset.get_preload_weight('iid')
        self.item_embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_item_emb.astype(np.float32)))
        self.item_embedding.weight.requires_grad = True

    def forward(self, item_seq, item_seq_len):
        embed = self.item_embedding(item_seq)

        output = self.mlp(embed)
        output = self.gather_indexes(output, item_seq_len - 1)  # [B H]
        return output  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)  # [B H]
            neg_items_emb = self.item_embedding(neg_items)  # [B H]
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores