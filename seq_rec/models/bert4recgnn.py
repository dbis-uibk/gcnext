import torch
import torch.nn as nn
import numpy as np

from recbole.model.sequential_recommender import BERT4Rec


class BERT4RecGNN(BERT4Rec):

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        pretrained_item_emb = dataset.get_preload_weight('iid')

        # bert4rec mask token embedding
        mask_emb = torch.FloatTensor(1, self.hidden_size)
        mask_emb.data.normal_(mean=0.0, std=self.initializer_range)

        emb_weights = torch.cat((torch.Tensor(pretrained_item_emb), torch.FloatTensor(1, self.hidden_size)))

        self.item_embedding = nn.Embedding.from_pretrained(emb_weights)
        self.item_embedding.weight.requires_grad = True
