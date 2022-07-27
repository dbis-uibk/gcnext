import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import itertools

from _operator import itemgetter
from math import sqrt, exp
import random
import time

import pandas as pd

from recbole.model.abstract_recommender import SequentialRecommender


class SKNNGNN(SequentialRecommender):

    def __init__(self, config, dataset):
        super(SKNNGNN, self).__init__(config, dataset)

        pretrained_item_emb = dataset.get_preload_weight('iid')
        self.item_embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_item_emb.astype(np.float32)))
        self.item_embedding.weight.requires_grad = False

        self.emb_threshold = config['emb_threshold']
        self.item_emb_dict = {}

        self.k = config['k']
        self.sample_size = config['sample_size']
        self.sampling = config['sampling']

        self.normalize = config['normalize']
        self.similarity = config['similarity']
        self.pop_boost = config['pop_boost']

        self.remind = config['remind']
        self.extend = config['extend']

        self.session_key = 'session_id'
        self.item_key = 'item_id'
        self.time_key = 'timestamp'
        self.item_list_key = 'item_list'

        # updated while recommending
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()

        # cache relations once at startup
        self.session_item_map = dict()
        self.item_session_map = dict()
        self.session_time = dict()

        all_item_ids = [i for i in range(len(dataset.field2id_token[self.ITEM_ID]))]
        self.item_ids = np.array(all_item_ids)

        self.fake_loss = torch.nn.Parameter(torch.zeros(1))

    def forward(self, item_seq):
        pass

    def calculate_loss(self, interaction):
        train = self.create_df_from_interaction(interaction.cpu())

        index_session = train.columns.get_loc(self.session_key)
        index_item = train.columns.get_loc(self.item_key)
        index_time = train.columns.get_loc(self.time_key)
        index_item_list = train.columns.get_loc(self.item_list_key)

        for i, row in enumerate(train.itertuples(index=False)):
            item_id_list = list(filter(lambda num: num != 0, row[index_item_list]))  # filter PAD id
            item_id_list.append(row[index_item])
            if row[index_session] in self.session_item_map:
                if len(item_id_list) > len(self.session_item_map.get(row[index_session])):
                    self.session_item_map[row[index_session]] = set(item_id_list)
                    self.session_time[row[index_session]] = row[index_time]
            else:
                self.session_item_map[row[index_session]] = set(item_id_list)
                self.session_time[row[index_session]] = row[index_time]

            for item_id in item_id_list:
                if item_id in self.item_session_map:
                    self.item_session_map[item_id].add(row[index_session])
                else:
                    self.item_session_map[item_id] = {row[index_session]}

        item_ids = torch.hstack((interaction[self.ITEM_SEQ], interaction[self.ITEM_ID][:, None]))
        item_ids = torch.unique(torch.flatten(item_ids, start_dim=0, end_dim=1))
        item_embs = self.item_embedding(item_ids)
        item_keys = item_ids.cpu().detach().numpy()
        zip_iterator = zip(item_keys, item_embs)
        self.item_emb_dict.update(zip_iterator)

        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        timestamp = interaction['timestamp']
        user = user.cpu().numpy().astype(int)
        item = item.cpu().numpy().astype(int)
        timestamp = timestamp.cpu().numpy().astype(int)
        result = []

        for index in range(len(user)):
            uid = user[index]
            iid = item[index]
            ts = timestamp[index]
            scores = self.predict_next(uid, iid, self.item_ids, ts)
            score = scores[iid]
            result.append(score)
        result = torch.from_numpy(np.array(result)).to(self.device)
        return result

    def full_sort_predict(self, interaction):
        batch_size = len(interaction[self.ITEM_ID])

        item_keys = interaction[self.ITEM_SEQ].flatten().unique()
        item_embs = self.item_embedding(item_keys)
        item_keys = item_keys.cpu().detach().numpy()
        zip_iterator = zip(item_keys, item_embs)
        current_item_embs = dict(zip_iterator)

        '''cos_sim = torch.cosine_similarity(item_embs.unsqueeze(dim=1).repeat(1, len(self.item_emb_dict.values()), 1),
                                torch.stack(list(self.item_emb_dict.values())), dim=2)'''

        self.current_item_emb_distances = {}
        stacked_item_embs = torch.stack(list(self.item_emb_dict.values()))
        for i, k in enumerate(set(item_keys)):
            k_sim = torch.cosine_similarity(current_item_embs[k], stacked_item_embs)
            self.current_item_emb_distances[k] = dict(zip(self.item_emb_dict.keys(), k_sim.cpu().numpy()))
 
        score = []
        for i in range(batch_size):
            uid = int(interaction[self.USER_ID][i])
            input_iid = int(interaction[self.ITEM_SEQ][i][interaction[self.ITEM_SEQ][i].nonzero()][-1])
            ts = int(interaction['timestamp'][i])
            score.append(self.predict_next(uid, input_iid, self.item_ids, ts))

        result = torch.from_numpy(np.array(score))
        return result

    def create_df_from_interaction(self, interaction):
        d = {'item_id': interaction.item_id, 'session_id': interaction.session_id,
             'timestamp': interaction.timestamp, 'item_list': interaction.item_id_list.tolist()}
        df = pd.DataFrame(data=d)
        return df

    def predict_next(self, session_id, input_item_id, predict_for_item_ids, timestamp=0, type='view'):

        if (self.session != session_id):  # new session

            if (self.extend):
                self.session_item_map[self.session] = self.session_items
                for item in self.session_items:
                    map_is = self.item_session_map.get(item)
                    if map_is is None:
                        map_is = set()
                        self.item_session_map.update({item: map_is})
                    map_is.add(self.session)

                ts = time.time()
                self.session_time.update({self.session: ts})

            self.session = session_id
            self.session_items = list()
            self.relevant_sessions = set()

        if type == 'view':
            self.session_items.append(input_item_id)

        neighbors = self.find_neighbors(set(self.session_items), input_item_id)
        scores = self.score_items(neighbors)

        # add some reminders
        if self.remind:

            reminderScore = 5
            takeLastN = 3

            cnt = 0
            for elem in self.session_items[-takeLastN:]:
                cnt = cnt + 1
                # reminderScore = reminderScore + (cnt/100)

                oldScore = scores.get(elem)
                newScore = 0
                if oldScore is None:
                    newScore = reminderScore
                else:
                    newScore = oldScore + reminderScore
                # print 'old score ', oldScore
                # update the score and add a small number for the position
                newScore = (newScore * reminderScore) + (cnt / 100)

                scores.update({elem: newScore})

        # push popular ones
        if self.pop_boost > 0:

            pop = self.item_pop(neighbors)
            # Iterate over the item neighbors
            # print itemScores
            for key in scores:
                item_pop = pop.get(key)
                # Gives some minimal MRR boost?
                scores.update({key: (scores[key] + (self.pop_boost * item_pop))})

        predictions = np.zeros(len(predict_for_item_ids))
        mask = np.in1d(predict_for_item_ids, list(scores.keys()))

        items = predict_for_item_ids[mask]
        values = [scores[x] for x in items]
        predictions[mask] = np.array(values)

        return predictions

    def item_pop(self, sessions):
        result = dict()
        max_pop = 0
        for session, weight in sessions:
            items = self.items_for_session(session)
            for item in items:

                count = result.get(item)
                if count is None:
                    result.update({item: 1})
                else:
                    result.update({item: count + 1})

                if (result.get(item) > max_pop):
                    max_pop = result.get(item)

        for key in result:
            result.update({key: (result[key] / max_pop)})

        return result

    def cosine_item_emb(self, current_items, neighbor_items):
        emb_distances = np.array([[self.current_item_emb_distances[c][n] for n in neighbor_items] for c in current_items])

        vp_sum = sum(emb_distances.flatten() >= self.emb_threshold)

        lneighbor = len(neighbor_items)
        current_sum = len(current_items)
        result = vp_sum / (sqrt(current_sum) * sqrt(lneighbor))

        return result

    def items_for_session(self, session):
        return self.session_item_map.get(session)

    def sessions_for_item(self, item_id):
        return self.item_session_map.get(item_id)

    def most_recent_sessions(self, sessions, number):
        sample = set()

        tuples = list()
        for session in sessions:
            time = self.session_time.get(session)
            if time is None:
                print(' EMPTY TIMESTAMP!! ', session)
            tuples.append((session, time))

        tuples = sorted(tuples, key=itemgetter(1), reverse=True)
        cnt = 0
        for element in tuples:
            cnt = cnt + 1
            if cnt > number:
                break
            sample.add(element[0])
        return sample

    def possible_neighbor_sessions(self, input_item_id):
        if self.sessions_for_item(input_item_id):
            self.relevant_sessions = self.sessions_for_item(input_item_id)
        else:
            self.relevant_sessions = []

        if self.sample_size == 0:  # use all session as possible neighbors

            return self.relevant_sessions

        else:  # sample some sessions

            if len(self.relevant_sessions) > self.sample_size:

                if self.sampling == 'recent':
                    sample = self.most_recent_sessions(self.relevant_sessions, self.sample_size)
                elif self.sampling == 'random':
                    sample = random.sample(self.relevant_sessions, self.sample_size)
                else:
                    sample = self.relevant_sessions[:self.sample_size]

                return sample
            else:
                return self.relevant_sessions

    def calc_similarity(self, session_items, sessions):
        neighbors = []
        cnt = 0
        for session in sessions:
            cnt = cnt + 1
            # get items of the session, look up the cache first
            session_items_test = self.items_for_session(session)

            similarity = self.cosine_item_emb(set(session_items), session_items_test)
            if similarity > 0:
                neighbors.append((session, similarity))

        return neighbors

    def find_neighbors(self, session_items, input_item_id):
        possible_neighbors = self.possible_neighbor_sessions(input_item_id)
        possible_neighbors = self.calc_similarity(session_items, possible_neighbors)

        possible_neighbors = sorted(possible_neighbors, reverse=True, key=lambda x: x[1])
        possible_neighbors = possible_neighbors[:self.k]

        return possible_neighbors

    def score_items(self, neighbors):
        scores = dict()
        for session in neighbors:
            items = self.items_for_session(session[0])

            for item in items:
                old_score = scores.get(item)
                new_score = session[1]

                if old_score is None:
                    scores.update({item: new_score})
                else:
                    new_score = old_score + new_score
                    scores.update({item: new_score})

        return scores