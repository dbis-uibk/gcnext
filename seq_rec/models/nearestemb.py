# -*- coding: utf-8 -*-
# @Time   : 2021/12/28
# @Author : Andreas Peintner
# @Email  : a.peintner@gmx.net

import numpy as np
import torch
import torch.nn as nn
from scipy import spatial

from recbole.model.abstract_recommender import SequentialRecommender


class NearestEmb(SequentialRecommender):

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.k = 100  # TODO config

        pretrained_item_emb = dataset.get_preload_weight('iid')
        self.item_embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_item_emb.astype(np.float32)))
        self.item_embedding.weight.requires_grad = False

        self.item_emb_tree = spatial.KDTree(self.item_embedding.weight)

        self.item_ids = np.array([i for i in range(len(dataset.field2id_token[self.ITEM_ID]))])

        self.fake_loss = torch.nn.Parameter(torch.zeros(1))

    def forward(self, item_seq):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        item = item.cpu().numpy().astype(int)
        result = []

        for index in range(len(user)):
            iid = item[index]
            scores = self.predict_next(iid)
            score = scores[iid]
            result.append(score)
        result = torch.from_numpy(np.array(result)).to(self.device)
        return result

    def full_sort_predict(self, interaction):
        batch_size = len(interaction[self.ITEM_ID])
        scores = []
        for i in range(batch_size):
            input_iid = interaction[self.ITEM_SEQ][i][interaction[self.ITEM_SEQ][i].nonzero()][-1]
            scores.append(self.predict_next(input_iid))

        result = torch.from_numpy(np.array(scores))
        return result

    def predict_next(self, input_iid):
        scores = self.score_nearest_items(input_iid)

        predictions = np.zeros(len(self.item_ids))
        mask = np.in1d(self.item_ids, list(scores.keys()))

        items = self.item_ids[mask]
        values = [scores[x] for x in items]
        predictions[mask] = np.array(values)

        return predictions

    def score_nearest_items(self, item_id):
        input_item_emb = self.item_embedding(item_id).cpu()
        dis, pos = self.item_emb_tree.query(input_item_emb, k=self.k)
        return dict(zip(pos[0], dis[0][::-1]))




