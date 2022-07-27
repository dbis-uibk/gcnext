import numpy as np
import scipy.sparse as sp
import torch

from _operator import itemgetter
from math import sqrt, exp, log10
import random
import time

import pandas as pd

from recbole.model.abstract_recommender import SequentialRecommender


class VSKNN(SequentialRecommender):

    def __init__(self, config, dataset):
        super(VSKNN, self).__init__(config, dataset)

        self.k = config['k']
        self.sample_size = config['sample_size']
        self.sampling = config['sampling']

        self.weighting = config['weighting']
        self.dwelling_time = config['dwelling_time']
        self.weighting_score = config['weighting_score']
        self.weighting_time = config['weighting_time']
        self.similarity = config['similarity']

        self.push_reminders = config['push_reminders']  # give more score to the items that belongs to the current session
        self.add_reminders = config['add_reminders']  # force the last 3 items of the current session to be in the top 20
        self.idf_weighting = config['idf_weighting']
        self.idf_weighting_session = config['idf_weighting_session']
        self.normalize = config['normalize']
        self.last_n_days = config['last_n_days']
        self.last_n_clicks = config['last_n_clicks']

        self.remind = config['remind']
        self.extend = config['extend'] # to add evaluated sessions to the maps

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

        if self.idf_weighting or self.idf_weighting_session:
            item_counter = dataset.item_counter
            keys, items = list(item_counter.keys()), list(item_counter.values())
            idf_items = np.log(len(dataset.user_counter) / np.array(items))
            self.idf = dict(zip(keys, idf_items))

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

            self.last_ts = -1
            self.session = session_id
            self.session_items = list()
            self.dwelling_times = list()
            self.relevant_sessions = set()

        if type == 'view':
            self.session_items.append(input_item_id)
            if self.dwelling_time:
                if self.last_ts > 0:
                    self.dwelling_times.append(timestamp - self.last_ts)
                self.last_ts = timestamp

        items = self.session_items if self.last_n_clicks is None else self.session_items[-self.last_n_clicks:]
        neighbors = self.find_neighbors(items, input_item_id, session_id, self.dwelling_times, timestamp)
        scores = self.score_items(neighbors, items, timestamp)

        predictions = np.zeros(len(predict_for_item_ids))
        mask = np.in1d(predict_for_item_ids, list(scores.keys()))

        items = predict_for_item_ids[mask]
        values = [scores[x] for x in items]
        predictions[mask] = np.array(values)
        series = pd.Series(data=predictions, index=predict_for_item_ids)

        if self.push_reminders:
            session_series = pd.Series(self.session_items)
            session_count = session_series.groupby(session_series).count() + 1

            series[session_count.index] *= session_count

        if self.add_reminders:
            session_series = pd.Series(index=self.session_items, data=series[self.session_items])
            session_series = session_series[session_series > 0]

            if len(session_series) > 0:
                session_series = session_series.iloc[:3]
                series.sort_values(ascending=False, inplace=True)
                session_series = session_series[session_series < series.iloc[19 - 3]]
                series[session_series.index] = series.iloc[19 - 3] + 1e-4

        if self.normalize:
            series = series / series.max()

        return series.values

    def item_pop(self, sessions):
        '''
        Returns a dict(item,score) of the item popularity for the given list of sessions (only a set of ids)

        Parameters
        --------
        sessions: set

        Returns
        --------
        out : dict
        '''
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

    def jaccard(self, first, second):
        '''
        Calculates the jaccard index for two sessions

        Parameters
        --------
        first: Id of a session
        second: Id of a session

        Returns
        --------
        out : float value
        '''
        sc = time.process_time()
        intersection = len(first & second)
        union = len(first | second)
        res = intersection / union

        self.sim_time += (time.process_time() - sc)

        return res

    def vec(self, current, neighbor, pos_map):
        '''
        Calculates the ? for 2 sessions

        Parameters
        --------
        first: Id of a session
        second: Id of a session

        Returns
        --------
        out : float value
        '''
        intersection = current & neighbor

        if pos_map is not None:
            vp_sum = 0
            current_sum = len(pos_map)
            for i in intersection:
                vp_sum += pos_map[i]

        else:
            vp_sum = len(intersection)
            current_sum = len(current)

        result = vp_sum / current_sum

        return result

    def cosine(self, current, neighbor, pos_map):
        '''
        Calculates the cosine similarity for two sessions

        Parameters
        --------
        first: Id of a session
        second: Id of a session

        Returns
        --------
        out : float value
        '''

        lneighbor = len(neighbor)
        intersection = current & neighbor

        if pos_map is not None:

            vp_sum = 0
            current_sum = 0
            for i in current:
                current_sum += pos_map[i] * pos_map[i]
                if i in intersection:
                    vp_sum += pos_map[i]
        else:
            vp_sum = len(intersection)
            current_sum = len(current)

        result = vp_sum / (sqrt(current_sum) * sqrt(lneighbor))

        return result

    def items_for_session(self, session):
        '''
        Returns all items in the session

        Parameters
        --------
        session: Id of a session

        Returns
        --------
        out : set
        '''
        return self.session_item_map.get(session);

    def vec_for_session(self, session):
        '''
        Returns all items in the session

        Parameters
        --------
        session: Id of a session

        Returns
        --------
        out : set
        '''
        return self.session_vec_map.get(session);

    def sessions_for_item(self, item_id):
        '''
        Returns all session for an item

        Parameters
        --------
        item: Id of the item session

        Returns
        --------
        out : set
        '''
        return self.item_session_map.get(item_id) if item_id in self.item_session_map else set()

    def most_recent_sessions(self, sessions, number):
        '''
        Find the most recent sessions in the given set

        Parameters
        --------
        sessions: set of session ids

        Returns
        --------
        out : set
        '''
        sample = set()

        tuples = list()
        for session in sessions:
            time = self.session_time.get(session)
            if time is None:
                print(' EMPTY TIMESTAMP!! ', session)
            tuples.append((session, time))

        tuples = sorted(tuples, key=itemgetter(1), reverse=True)
        # print 'sorted list ', sortedList
        cnt = 0
        for element in tuples:
            cnt = cnt + 1
            if cnt > number:
                break
            sample.add(element[0])
        # print 'returning sample of size ', len(sample)
        return sample

    def possible_neighbor_sessions(self, session_items, input_item_id, session_id):
        '''
        Find a set of session to later on find neighbors in.
        A self.sample_size of 0 uses all sessions in which any item of the current session appears.
        self.sampling can be performed with the options "recent" or "random".
        "recent" selects the self.sample_size most recent sessions while "random" just choses randomly.

        Parameters
        --------
        sessions: set of session ids

        Returns
        --------
        out : set
        '''

        self.relevant_sessions = self.relevant_sessions | self.sessions_for_item(input_item_id)

        if self.sample_size == 0:  # use all session as possible neighbors

            print('!!!!! runnig KNN without a sample size (check config)')
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

    def calc_similarity(self, session_items, sessions, dwelling_times, timestamp):
        '''
        Calculates the configured similarity for the items in session_items and each session in sessions.

        Parameters
        --------
        session_items: set of item ids
        sessions: list of session ids

        Returns
        --------
        out : list of tuple (session_id,similarity)
        '''

        pos_map = {}
        length = len(session_items)

        count = 1
        for item in session_items:
            if self.weighting is not None:
                pos_map[item] = getattr(self, self.weighting)(count, length)
                count += 1
            else:
                pos_map[item] = 1

        if self.dwelling_time:
            dt = dwelling_times.copy()
            dt.append(0)
            dt = pd.Series(dt, index=session_items)
            dt = dt / dt.max()
            # dt[session_items[-1]] = dt.mean() if len(session_items) > 1 else 1
            dt[session_items[-1]] = 1

            # print(dt)
            for i in range(len(dt)):
                pos_map[session_items[i]] *= dt.iloc[i]
            # print(pos_map)

        if self.idf_weighting_session:
            max = -1
            for item in session_items:
                pos_map[item] = self.idf[item] if item in self.idf else 0
        #                 if pos_map[item] > max:
        #                     max = pos_map[item]
        #             for item in session_items:
        #                 pos_map[item] = pos_map[item] / max

        # print 'nb of sessions to test ', len(sessionsToTest), ' metric: ', self.metric
        items = set(session_items)
        neighbors = []
        cnt = 0
        for session in sessions:
            cnt = cnt + 1
            # get items of the session, look up the cache first
            n_items = self.items_for_session(session)
            sts = self.session_time[session]

            # dot product
            # similarity = self.vec(items, n_items, pos_map)
            similarity = getattr(self, self.similarity)(items, n_items, pos_map)

            if similarity > 0:

                if self.weighting_time:
                    diff = timestamp - sts
                    days = round(diff / 60 / 60 / 24)
                    decay = pow(7 / 8, days)
                    similarity *= decay

                # print("days:",days," => ",decay)

                neighbors.append((session, similarity))

        return neighbors

    # -----------------
    # Find a set of neighbors, returns a list of tuples (sessionid: similarity)
    # -----------------
    def find_neighbors(self, session_items, input_item_id, session_id, dwelling_times, timestamp):
        '''
        Finds the k nearest neighbors for the given session_id and the current item input_item_id.

        Parameters
        --------
        session_items: set of item ids
        input_item_id: int
        session_id: int

        Returns
        --------
        out : list of tuple (session_id, similarity)
        '''
        possible_neighbors = self.possible_neighbor_sessions(session_items, input_item_id, session_id)
        possible_neighbors = self.calc_similarity(session_items, possible_neighbors, dwelling_times, timestamp)

        possible_neighbors = sorted(possible_neighbors, reverse=True, key=lambda x: x[1])
        possible_neighbors = possible_neighbors[:self.k]

        return possible_neighbors

    def score_items(self, neighbors, current_session, timestamp):
        '''
        Compute a set of scores for all items given a set of neighbors.

        Parameters
        --------
        neighbors: set of session ids

        Returns
        --------
        out : list of tuple (item, score)
        '''
        # now we have the set of relevant items to make predictions
        scores = dict()
        iset = set(current_session)
        # iterate over the sessions
        for session in neighbors:
            # get the items in this session
            items = self.items_for_session(session[0])
            step = 1

            for item in reversed(current_session):
                if item in items:
                    decay = getattr(self, self.weighting_score + '_score')(step)
                    break
                step += 1

            for item in items:

                if not self.remind and item in iset:
                    continue

                old_score = scores.get(item)
                new_score = session[1]

                if self.idf_weighting and item in self.idf:
                    new_score = new_score + (new_score * self.idf[item] * self.idf_weighting)

                new_score = new_score * decay

                if not old_score is None:
                    new_score = old_score + new_score

                scores.update({item: new_score})

        return scores

    def linear_score(self, i):
        return 1 - (0.1 * i) if i <= 100 else 0

    def same_score(self, i):
        return 1

    def div_score(self, i):
        return 1 / i

    def log_score(self, i):
        return 1 / (log10(i + 1.7))

    def quadratic_score(self, i):
        return 1 / (i * i)

    def linear(self, i, length):
        return 1 - (0.1 * (length - i)) if i <= 10 else 0

    def same(self, i, length):
        return 1

    def div(self, i, length):
        return i / length

    def log(self, i, length):
        return 1 / (log10((length - i) + 1.7))

    def quadratic(self, i, length):
        return (i / length) ** 2