import torch
import torch.nn as nn
import numpy as np

from recbole.model.sequential_recommender import GRU4Rec


class GRU4RecGNN(GRU4Rec):

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        pretrained_item_emb = dataset.get_preload_weight('iid')
        self.item_embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_item_emb.astype(np.float32)))

        self.item_embedding.weight.requires_grad = True
