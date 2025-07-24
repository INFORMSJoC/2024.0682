import gc
import json
import os
import re
from time import time
import math
import numpy as np
import pandas as pd
from scipy.io import loadmat
import scipy.sparse as sp
import torch
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from collections import Counter
import world
from sampler import sample_from_hist, sample_from_nbr
from utils import calculate_metrics


def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))


class PairDataset:
    def __init__(self, src="Ciao"):
        self.src = src
        self.dataset_name = f'{src}'
        try:
            self.train_set = pd.read_csv(os.path.join(world.DATA_PATH, 'preprocessed', self.dataset_name, 'train_set.txt'))
            self.test_set = pd.read_csv(os.path.join(world.DATA_PATH, 'preprocessed', self.dataset_name, 'test_set.txt'))
            self.item_popularity = pd.read_csv(os.path.join(world.DATA_PATH, 'preprocessed', self.dataset_name, 'item_popularity.txt'))
            self.n_user = pd.concat([self.train_set, self.test_set])['user'].nunique()
            self.m_item = pd.concat([self.train_set, self.test_set])['item'].nunique()
        except IOError:
            interactionNet, self.n_user, self.m_item = loadInteraction(src, self.dataset_name,
                                                                            prepro=world.prepro, posThreshold=0,
                                                                            filter_social=2, delete_ratio=world.delete_ratio)
            self.train_set, self.test_set = split_dataset_by_time(interactionNet)
            self.item_popularity = compute_popularity(self.train_set, tau_p=1 * pow(10, 7))
            self.train_set.to_csv(os.path.join(world.DATA_PATH, 'preprocessed', self.dataset_name, 'train_set.txt'), index=False)
            self.test_set.to_csv(os.path.join(world.DATA_PATH, 'preprocessed', self.dataset_name, 'test_set.txt'), index=False)
            self.item_popularity.to_csv(os.path.join(world.DATA_PATH, 'preprocessed', self.dataset_name, 'item_popularity.txt'), index=False)

        interactionNet = pd.concat([self.train_set,self.test_set])
        self.niche_items = calculate_metrics(interactionNet)
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++')

        test_data_satisfactory = self.test_set.loc[self.test_set['rating'] > 4].reset_index(drop=True)
        train_data_satisfactory = self.train_set.loc[self.train_set['rating'] > 4].reset_index(drop=True)
        satis_num = len(test_data_satisfactory) + len(train_data_satisfactory)

        self.train_set, self.val_set = splitDataset(self.train_set, 'fo', testSize=0.1)
        # self.val_set, self.test_set = split_val_test(self.test_set, test_size=0.7)
        self.trainUser = np.array(self.train_set['user'])
        self.trainUniqueUser = np.unique(self.train_set['user'])
        self.trainItem = np.array(self.train_set['item'])
        self.trainRating = np.array(self.train_set['rating'])
        self.trainTimestamp = np.array(self.train_set['timestamp'])
        self._trainDataSize = len(self.train_set)
        self._valDataSize = len(self.val_set)
        self._testDataSize = len(self.test_set)

        self.max_time = self.train_set['timestamp'].max()

        self._trainUserNum, self._valUserNum, self._testUserNum = len(np.unique(self.train_set['user'])),\
                                                                  len(np.unique(self.val_set['user'])),\
                                                                  len(np.unique(self.test_set['user']))
        print(f"{self._trainDataSize}, {self._valDataSize}, {self._testDataSize} interactions in train, val, test  set")
        print(f"{self._trainUserNum}, {self._valUserNum}, {self._testUserNum} users in train, val, test  set")
        print(f"Number of users: {self.n_user}\n Number of items: {self.m_item}")
        print(f"Number of Ratings: {self._trainDataSize + self._testDataSize}")
        print(f"{world.dataset} Rating Sparsity: {1-(self._trainDataSize + self._valDataSize + self._testDataSize)*100 / self.n_user / self.m_item}")
        print(f"satisfactions: {satis_num}, satisfaction rate:{satis_num/(self._trainDataSize + self._valDataSize + self._testDataSize)}")

        self.item_counts = load_item_counts(self.train_set, self.m_item)
        self.test_data_satisfactory = self.test_set.loc[self.test_set['rating'] > 4].reset_index(drop=True)
        self.longtail_items = get_longtail_items(self.train_set, self.m_item)

        self.interactionGraph = None
        self.UserItemNet = csr_matrix((np.ones(len(self.train_set)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))


        #  user's history interacted items
        self._allPos = self.getUserPosItems(list(range(self.n_user)),training=True)
        # get val dictionary
        self._valDic = self.__build_val()
        # get test dictionary
        self._testDic = self.__build_test()
        self._satisfactoryTestDic = self.__build_satisfactory_test()
        self._coldTestDic = self.__build_cold_test()
        self._userDic, self._itemDic = self._getInteractionDic()


    @property
    def userDic(self):
        return self._userDic

    @property
    def itemDic(self):
        return self._itemDic

    @property
    def valDict(self):
        return self._valDic

    @property
    def testDict(self):
        return self._testDic

    @property
    def coldTestDict(self):
        return self._coldTestDic

    @property
    def satisfactoryTestDict(self):
        return self._satisfactoryTestDic

    @property
    def allPos(self):
        return self._allPos

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self._trainDataSize

    def getUserPosItems(self, users, training=True):
        """
        Method of get user all positive items
        Returns
        -------
        [ndarray0,...,ndarray_users]
        """
        if training:
            interaction_data = self.train_set
        else:
            interaction_data = pd.concat([self.train_set,self.val_set]).reset_index()
        posItems = []
        for user in users:
            item_u = interaction_data[interaction_data['user']==user]['item'].values
            posItems.append(item_u)
        return posItems

    def __build_val(self):
        """
        Method of build test dictionary
        Returns
        -------
            dict: {user: [items]}
        """
        val_data = {}
        for i in range(len(self.val_set)):
            user = self.val_set['user'][i]
            item = self.val_set['item'][i]
            if val_data.get(user):
                val_data[user].append(item)
            else:
                val_data[user] = [item]
        return val_data


    def __build_test(self):
        """
        Method of build test dictionary
        Returns
        -------
            dict: {user: [items]}
        """
        test_data = {}
        for i in range(len(self.test_set)):
            user = self.test_set['user'][i]
            item = self.test_set['item'][i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def __build_cold_test(self):
        test_data = self._testDic.copy()
        for i in list(test_data.keys()):
            try:
                if self.train_set['user'].value_counts()[i] > 20:
                    del test_data[i]
            except:
                pass
        return test_data

    def __build_satisfactory_test(self):
        test_data = {}
        for i in range(len(self.test_data_satisfactory)):
            user = self.test_data_satisfactory['user'][i]
            item = self.test_data_satisfactory['item'][i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data


    def _getInteractionDic(self):
        user_interaction = {}
        item_interaction = {}

        def getDict(_set):
            for i in range(len(_set)):
                user = _set['user'][i]
                item = _set['item'][i]
                if user_interaction.get(user):
                    user_interaction[user].append(item)
                else:
                    user_interaction[user] = [item]
                if item_interaction.get(item):
                    item_interaction[item].append(user)
                else:
                    item_interaction[item] = [user]

        getDict(self.train_set)
        getDict(self.val_set)
        getDict(self.test_set)
        return user_interaction, item_interaction


class GraphDataset(PairDataset):
    def __init__(self, src):
        super(GraphDataset, self).__init__(src)
        # build (users,items), bipartite graph
        self.interactionGraph = None


    def getInteractionGraph(self):
        if self.interactionGraph is None:
            try:
                norm_adj = sp.load_npz(os.path.join(world.DATA_PATH, 'preprocessed', self.src, 'interaction_adj_mat.npz'))
                print("successfully loaded normalized interaction adjacency matrix")
            except IOError:
                print("generating adjacency matrix")
                start = time()
                adj_mat = sp.dok_matrix((self.n_user + self.m_item, self.n_user + self.m_item), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_user, self.n_user:] = R
                adj_mat[self.n_user:, :self.n_user] = R.T
                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                with np.errstate(divide='ignore'):
                    d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                sp.save_npz(os.path.join(world.DATA_PATH, 'preprocessed', self.dataset_name, 'interaction_adj_mat.npz'), norm_adj)
                print(f"costing {time() - start}s, saved normalized interaction adjacency matrix")

            self.interactionGraph = _convert_sp_mat_to_sp_tensor(norm_adj)
            self.interactionGraph = self.interactionGraph.coalesce().to(world.device)
        return self.interactionGraph



class SocialGraphDataset(GraphDataset):
    def __init__(self, src):
        super(SocialGraphDataset, self).__init__(src)
        self.friendNet = loadFriend(src, self.dataset_name)
        self.socialNet = csr_matrix((np.ones(len(self.friendNet)), (self.friendNet['user'], self.friendNet['trust'])),
                                    shape=(self.n_user, self.n_user))
        self.socialGraph = None

        print(f"Number of Links: {len(self.friendNet)}")
        print(f"{world.dataset} Link Density: {(1-len(self.friendNet) / self.n_user / self.n_user)*100}")
        self._allFriends = self.getUserFriends(list(range(self.n_user)))


    def getSocialGraph(self):
        if self.socialGraph is None:
            try:
                pre_adj_mat = sp.load_npz(os.path.join(world.DATA_PATH, 'preprocessed', self.dataset_name, 'social_adj_mat.npz'))
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except IOError:
                print("generating adjacency matrix")
                start = time()
                adj_mat = self.socialNet.tolil()
                rowsum = np.array(adj_mat.sum(axis=1))
                with np.errstate(divide='ignore'):
                    d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                print(f"costing {time() - start}s, saved norm_mat...")
                sp.save_npz(os.path.join(world.DATA_PATH, 'preprocessed', self.dataset_name, 'social_adj_mat.npz'), norm_adj)

            self.socialGraph = _convert_sp_mat_to_sp_tensor(norm_adj)
            self.socialGraph = self.socialGraph.coalesce().to(world.device)
        return self.socialGraph



    def getDenseSocialGraph(self):
        if self.socialGraph is None:
            self.socialGraph = self.getSocialGraph().to_dense()
        else:
            pass
        return self.socialGraph

    @property
    def AllFriends(self):
        return self._allFriends

    def getUserFriends(self, users):
        """
        Method of get user all friends
        Returns
        -------
        [ndarray0,...,ndarray_users]
        """
        posFriends = []
        for user in users:
            posFriends.append(self.socialNet[user].nonzero()[1])
        return posFriends

    def get_nbrs(self):
        all_nbrs = self.AllFriends
        u_nbrs = np.zeros((self.n_user, world.config['max_nbrslen']))
        for u in range(self.n_user):
            u_friend = sample_from_nbr(all_nbrs[int(u)], world.config['max_nbrslen'],
                                            masked_value=self.n_user, input_user=int(u))
            u_nbrs[u] = u_friend
        return u_nbrs

    def get_hists(self):
        all_pos = self.allPos
        u_hists = np.zeros((self.n_user, world.config['max_histslen']))
        for u in range(self.n_user):
            u_hist = sample_from_hist(all_pos[int(u)], world.config['max_histslen'],
                                            masked_value=self.m_item)
            u_hists[u] = u_hist
        return u_hists

    def get_nbrs_hists(self):
        all_pos = self.allPos
        all_nbrs = self.AllFriends
        u_nbrs = np.zeros((self.n_user, world.config['max_nbrslen']))
        u_nbrs_hists = np.zeros((self.n_user, world.config['max_nbrslen'], world.config['max_histslen']))
        for u in range(self.n_user):
            u_friend = sample_from_nbr(all_nbrs[int(u)], world.config['max_nbrslen'],
                                            masked_value=self.n_user, input_user=int(u))
            u_nbrs[u] = u_friend

            for j, friend in enumerate(u_friend):
                if friend == self.n_user:
                    item = np.zeros((world.config['max_histslen']), dtype=int)+self.m_item
                else:
                    item = sample_from_hist(all_pos[int(j)], world.config['max_histslen'],
                                            masked_value=self.m_item)
                u_nbrs_hists[u, j] = item
        return u_nbrs, u_nbrs_hists



class DecGraphDataset(SocialGraphDataset):
    def __init__(self, src):
        super(DecGraphDataset, self).__init__(src)
        # self.item_popularity_train = self.item_popularity.iloc[:, :-1]
        user_counts = Counter(self.friendNet['trust'].values)
        for u in range(self.n_user):
            if u not in user_counts.keys():
                user_counts[u] = 0
        self.degree_df = pd.DataFrame(list(user_counts.items()), columns=['user', 'degree'])

    def bucketize_items(self):
        if world.config['pop_fading']:
            temp_df = self.item_popularity.copy()
            temp_df = temp_df.groupby('item')['popularity'].mean()
        else:
            item_counts = Counter(self.train_set['item'].values)
            for i in range(self.m_item):
                if i not in item_counts.keys():
                    item_counts[i] = 0
            temp_df = pd.DataFrame(list(item_counts.items()), columns=['item', 'popularity'])

        item_popularity = pd.DataFrame(index=np.arange(self.m_item))
        item_popularity = pd.merge(item_popularity, temp_df, left_index=True, right_index=True, how='left').fillna(0)

        item_popularity, bucket_dists = self.bucketize(item_popularity, 'popularity', world.config['pop_num'])
        item_popularity = item_popularity.reset_index()
        item_popularity = item_popularity.sort_values('index')
        bucket_ids = item_popularity[['bucket_popularity']].values.flatten()
        return np.array(bucket_ids), np.array(bucket_dists)

    def get_pop_items(self):
        temp_df = self.item_popularity_train.copy()
        temp_df = temp_df.sort_index()
        temp_df = temp_df.apply(lambda x: x/x.max(), axis=0)
        item_K_pops = temp_df.values.flatten()
        return np.array(item_K_pops)

    def bucketize_users(self):
        temp_df = self.degree_df
        user_gini = gini_coefficient(list(Counter(temp_df['degree'].values)))
        print(f'user_gini:{user_gini}')
        temp_df, bucket_dists = self.bucketize(temp_df, 'degree', world.config['degree_num'])
        temp_df = temp_df.sort_values('user')
        bucket_ids = temp_df[['bucket_degree']].values.flatten()
        return np.array(bucket_ids), np.array(bucket_dists)

    def bucketize(self, data, col, n):
        sorted_df = data.sort_values(by=col)
        # sorted_df['bucket_'+col] = pd.cut(sorted_df[col], n)
        sorted_df['bucket_' + col] = pd.qcut(sorted_df[col], n, duplicates='drop')
        sorted_df = sorted_df.sort_values(by=['bucket_' + col])
        sorted_df['bucket_'+col] = pd.Categorical(sorted_df['bucket_'+col]).codes
        sorted_df['bucket_'+col] = pd.Categorical(sorted_df['bucket_'+col]).codes
        mean_buckets = sorted_df.groupby(by='bucket_'+col)[col].mean().values
        normalized_means = mean_buckets / mean_buckets.sum()
        return sorted_df, normalized_means


def compute_popularity(train_data, tau_p):
    train_data = train_data.copy()
    train_data['popularity'] = 0.0

    for item, group in train_data.groupby('item'):
        # train_data
        group = group.sort_values('timestamp')

        timestamps = group['timestamp'].values

        item_popularity = np.zeros(len(timestamps))

        for j in range(1, len(timestamps)):
            delta_time = timestamps[j] - timestamps[j - 1]
            item_popularity[j] = (item_popularity[j - 1] + 1) * np.exp(-delta_time / tau_p)

        train_data.loc[group.index, 'popularity'] = item_popularity

    return train_data


def loadInteraction(src='Ciao', dataset_name='', prepro='origin', binary=False, posThreshold=None, level='ui',
                    filter_social=2, delete_ratio=0):
    """
        Method of loading certain raw data
        Parameters
        ----------
        src : str, the name of dataset
        prepro : str, way to pre-process raw data input, expect 'origin', f'{N}core', f'{N}filter', N is integer value
        binary : boolean, whether to transform rating to binary label as CTR or not as Regression
        posThreshold : float, if not None, treat rating larger than this threshold as positive sample
        level : str, which level to do with f'{N}core' or f'{N}filter' operation
        (it only works when prepro contains 'core' or 'filter')

        Returns
        -------
        df : pd.DataFrame, rating information with columns: user, item, rating, (options: timestamp)
        user_num : int, the number of users
        item_num : int, the number of items
        """
    # which dataset will use
    if src == 'Ciao':
        rating_mat = loadmat(os.path.join(world.DATA_PATH, 'raw', src, 'rating_with_timestamp.mat'))
        df = pd.DataFrame(data=rating_mat['rating'], columns=['user', 'item', 'cate', 'rating', 'help', 'timestamp'])
        del rating_mat, df['cate'], df['help']
        gc.collect()

    elif src == 'Epinions':
        rating_mat = loadmat(os.path.join(world.DATA_PATH, 'raw', src, 'rating_with_timestamp.mat'))
        df = pd.DataFrame(data=rating_mat['rating_with_timestamp'],
                          columns=['user', 'item', 'cate', 'rating', 'help', 'timestamp'])
        del rating_mat, df['cate'], df['help']
        gc.collect()

    elif src == 'Philadelphia' or src == 'Tucson':
        df = pd.read_csv(os.path.join(world.DATA_PATH, 'raw', src, 'ratings.csv'))
        df['timestamp'] = pd.to_datetime(df['timestamp']).view('int64')// 10**9
    else:
        raise ValueError('Invalid Dataset Error')

    item_support = df['item'].value_counts()
    sorted_items = item_support.index
    num_items_to_remove = int(len(sorted_items) * delete_ratio)
    items_to_keep = sorted_items[num_items_to_remove:]
    df = df[df['item'].isin(items_to_keep)]


    if filter_social:
        trust_df = readFriend(src)
        user_counts = trust_df['user'].value_counts() + trust_df['trust'].value_counts()
        social_users = user_counts[user_counts >= filter_social].index
        b_users = len(df['user'].unique())
        df = df[df['user'].isin(social_users)]
        f_users = len(df['user'].unique())
        deleted_users = b_users - f_users
        print(f'[Delete] {deleted_users}')

    # set rating >= threshold as positive samples
    if posThreshold is not None:
        df = df.query(f'rating >= {posThreshold}').reset_index(drop=True)

    # reset rating to interaction, here just treat all rating as 1
    if binary:
        df['rating'] = 1.0

    # which type of pre-dataset will use
    if prepro == 'origin':
        pass

    elif prepro.endswith('filter'):
        pattern = re.compile(r'\d+')
        filter_num = int(pattern.findall(prepro)[0])

        # count user's item
        tmp1 = df.groupby(['user'], as_index=False)['item'].count()
        tmp1.rename(columns={'item': 'cnt_item'}, inplace=True)
        # count item's user
        tmp2 = df.groupby(['item'], as_index=False)['user'].count()
        tmp2.rename(columns={'user': 'cnt_user'}, inplace=True)
        # add column of count numbers
        df = df.merge(tmp1, on=['user']).merge(tmp2, on=['item'])
        if level == 'ui':
            df = df.query(f'cnt_item >= {filter_num} and cnt_user >= {filter_num}').reset_index(drop=True)
        elif level == 'u':
            df = df.query(f'cnt_item >= {filter_num}').reset_index(drop=True)
        elif level == 'i':
            df = df.query(f'cnt_user >= {filter_num}').reset_index(drop=True)
        else:
            raise ValueError(f'Invalid level value: {level}')

        df.drop(['cnt_item', 'cnt_user'], axis=1, inplace=True)
        del tmp1, tmp2
        gc.collect()

    elif prepro.endswith('core'):
        pattern = re.compile(r'\d+')
        core_num = int(pattern.findall(prepro)[0])

        def filter_user(df):
            tmp = df.groupby(['user'], as_index=False)['item'].count()
            tmp.rename(columns={'item': 'cnt_item'}, inplace=True)
            df = df.merge(tmp, on=['user'])
            df = df.query(f'cnt_item >= {core_num}').reset_index(drop=True).copy()
            df.drop(['cnt_item'], axis=1, inplace=True)

            return df

        def filter_item(df):
            tmp = df.groupby(['item'], as_index=False)['user'].count()
            tmp.rename(columns={'user': 'cnt_user'}, inplace=True)
            df = df.merge(tmp, on=['item'])
            df = df.query(f'cnt_user >= {core_num}').reset_index(drop=True).copy()
            df.drop(['cnt_user'], axis=1, inplace=True)

            return df

        if level == 'ui':
            while 1:
                df = filter_user(df)
                df = filter_item(df)
                chk_u = df.groupby('user')['item'].count()
                chk_i = df.groupby('item')['user'].count()
                if len(chk_i[chk_i < core_num]) <= 0 and len(chk_u[chk_u < core_num]) <= 0:
                    break
        elif level == 'u':
            df = filter_user(df)
        elif level == 'i':
            df = filter_item(df)
        else:
            raise ValueError(f'Invalid level value: {level}')

        gc.collect()

    else:
        raise ValueError('Invalid dataset preprocess type, origin/Ncore/Nfilter (N is int number) expected')

    # encoding user_id and item_id
    userId = pd.Categorical(df['user'])
    itemId = pd.Categorical(df['item'])
    df['user'] = userId.codes
    df['item'] = itemId.codes

    userCodeDict = {int(value): code for code, value in enumerate(userId.categories.values)}
    itemCodeDict = {int(value): code for code, value in enumerate(itemId.categories.values)}

    outputPath = os.path.join(world.DATA_PATH, 'preprocessed', dataset_name)
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    with open(os.path.join(outputPath, 'userReindex.json'), "w") as f:
        f.write(json.dumps(userCodeDict))
    with open(os.path.join(outputPath, 'itemReindex.json'), "w") as f:
        f.write(json.dumps(itemCodeDict))

    userNum = df['user'].nunique()
    itemNum = df['item'].nunique()

    print(f'Finish loading [{src}]-[{prepro}] dataset')

    return df, userNum, itemNum


def readFriend(src):
    if src == 'Ciao'or src == 'Epinions':
        uu_elist = loadmat(os.path.join(world.DATA_PATH, 'raw', src, 'trust.mat'))['trust']
        friendNet = pd.DataFrame(uu_elist, columns=['user', 'trust'])
        del uu_elist
        gc.collect()

    elif src == 'Philadelphia' or src == 'Tucson':
        friendNet = pd.read_csv(os.path.join(world.DATA_PATH, 'raw', src, 'trust.csv'))
    else:
        raise ValueError('Invalid Dataset Error')

    gc.collect()
    friendNet = friendNet[friendNet['user'] != friendNet['trust']].reset_index(drop=True)
    return friendNet


def loadFriend(src, dataset_name):
    path = os.path.join(world.DATA_PATH, 'preprocessed', dataset_name, 'trust.txt')
    if os.path.exists(path):
        return pd.read_csv(path)

    friendNet = readFriend(src)
    friendNet = renameFriendID(dataset_name, friendNet)
    friendNet.to_csv(path, index=False)
    return friendNet


def renameFriendID(src: str, friendNet: pd.DataFrame):
    with open(os.path.join(world.DATA_PATH, 'preprocessed', src, 'userReindex.json')) as f:
        userReindex = json.load(f)
    friendNet['user'] = friendNet.apply(lambda x: reIndex(x.user, userReindex), axis=1)
    friendNet['trust'] = friendNet.apply(lambda x: reIndex(x.trust, userReindex), axis=1)
    friendNet = friendNet.drop(friendNet[(friendNet['user'] == -1) | (friendNet['trust'] == -1)].index)
    return friendNet


def reIndex(x, userReindex):
    if str(x) in userReindex.keys():
        return userReindex[str(x)]
    else:
        return -1

def split_dataset_by_time(data):
    min_time = data['timestamp'].min()
    max_time = data['timestamp'].max()
    K = 10
    time_interval = math.ceil((max_time - min_time) / K)
    train_time_max = min_time + time_interval * (K - 1)
    train_data = data[data['timestamp'] < train_time_max]
    test_data = data[data['timestamp'] >= train_time_max].reset_index(drop=True)
    return train_data, test_data

def compute_K_item_popularity(data, item_num):
    min_time = data['timestamp'].min()
    max_time = data['timestamp'].max()
    K = 10
    time_interval = math.ceil((max_time - min_time) / K)
    item_popularity = pd.DataFrame(index=np.arange(item_num))
    data_concat = pd.DataFrame()
    for i in range(K):
        time_split_min = min_time + time_interval * i
        time_split_max = time_split_min + time_interval
        if i == K - 1:
            time_split_max = data['timestamp'].max() + 1
        data_split = data[
            (data['timestamp'] >= time_split_min) & (data['timestamp'] < time_split_max)].reset_index(drop=True)
        # print(time_split_min, time_split_max, len(data_split))
        # item_counts = Counter(data_split['item'].values)
        # item_counts_df = pd.DataFrame(list(item_counts.items()), columns=['item', 'item_pop'])
        count = data_split.item.value_counts()
        pop = pd.DataFrame(count)
        pop.columns = [str(i)]
        # (pop, pop.values.max())
        # pop = pop / np.max(pop.values)
        item_popularity = pd.merge(item_popularity, pop, left_index=True, right_index=True, how='left').fillna(0)
        data_split['split_idx'] = i
        data_concat = pd.concat([data_concat, data_split]).reset_index(drop=True)
    all_data = data_concat
    # popId = pd.Categorical(all_data['item_pop'])
    # all_data['item_pop'] = popId.codes
    return all_data, item_popularity


def splitDataset(df, testMethod='tfo', testSize=.2):
    """
    this is from https://github.com/recsys-benchmark/DaisyRec-v2.0
    method of splitting data into training data and test data
    Parameters
    ----------
    df : pd.DataFrame raw data waiting for test set splitting
    testMethod : str, way to split test set
                    'fo': split by ratio
                    'tfo': split by ratio with timestamp
                    'tloo': leave one out with timestamp
                    'loo': leave one out
                    'ufo': split by ratio in user level
                    'utfo': time-aware split by ratio in user level
    testSize : float, size of test set
    valSize : float, size of validate set

    Returns
    -------
    train_set : pd.DataFrame training dataset
    test_set : pd.DataFrame test dataset

    """

    train_set, test_set = pd.DataFrame(), pd.DataFrame()
    if testMethod == 'ufo':
        driver_ids = df['user']
        _, driver_indices = np.unique(np.array(driver_ids), return_inverse=True)
        gss = GroupShuffleSplit(n_splits=1, test_size=testSize, random_state=42)
        for train_idx, test_idx in gss.split(df, groups=driver_indices):
            train_set, test_set = df.loc[train_idx, :].copy(), df.loc[test_idx, :].copy()

    elif testMethod == 'utfo':
        df = df.sort_values(['user', 'timestamp']).reset_index(drop=True)

        def time_split(grp):
            start_idx = grp.index[0]
            split_len = int(np.ceil(len(grp) * (1 - testSize)))
            split_idx = start_idx + split_len
            end_idx = grp.index[-1]
            return list(range(split_idx, end_idx + 1))

        test_index = df.groupby('user').apply(time_split).explode().values
        test_set = df[df.index.isin(test_index)].copy()
        train_set = df[~df.index.isin(test_index)].copy()

    elif testMethod == 'tfo':
        # df = df.sample(frac=1)
        df = df.sort_values(['timestamp']).reset_index(drop=True)
        split_idx = int(np.ceil(len(df) * (1 - testSize)))
        train_set, test_set = df.iloc[:split_idx, :].copy(), df.iloc[split_idx:, :].copy()

    elif testMethod == 'fo':
        train_set, test_set = train_test_split(df, test_size=testSize, random_state=42)

    elif testMethod == 'tloo':
        # df = df.sample(frac=1)
        df = df.sort_values(['timestamp']).reset_index(drop=True)
        df['rank_latest'] = df.groupby(['user'])['timestamp'].rank(method='first', ascending=False)
        train_set, test_set = df[df['rank_latest'] > 1].copy(), df[df['rank_latest'] == 1].copy()
        del train_set['rank_latest'], test_set['rank_latest']

    elif testMethod == 'loo':
        # # slow method
        # test_set = df.groupby(['user']).apply(pd.DataFrame.sample, n=1).reset_index(drop=True)
        # test_key = test_set[['user', 'item']].copy()
        # train_set = df.set_index(['user', 'item']).drop(pd.MultiIndex.from_frame(test_key)).reset_index().copy()

        # # quick method
        test_index = df.groupby(['user']).apply(lambda grp: np.random.choice(grp.index))
        test_set = df.loc[test_index, :].copy()
        train_set = df[~df.index.isin(test_index)].copy()

    else:
        raise ValueError('Invalid data_split value, expect: loo, fo, tloo, tfo')

    train_set, test_set = train_set.reset_index(drop=True), test_set.reset_index(drop=True)
    # print(f'train interaction num:{len(train_set)}, test interaction num:{len(test_set)}')
    return train_set, test_set


def get_longtail_items(df, num_items):
    item_counts = Counter(df['item'].values)
    for i in range(num_items):
        if i not in item_counts.keys():
            item_counts[i] = 0
    item_counts_df = pd.DataFrame(list(item_counts.items()), columns=['item', 'count'])
    item_counts_df = item_counts_df.sort_values(by='count', ascending=False).reset_index(drop=True)
    threshold_20 = int(len(item_counts_df) * 0.2)
    longtail_items = item_counts_df[threshold_20:]['item'].values
    longtail_items = set(longtail_items)
    return longtail_items


def gini_coefficient(x):
    x = np.array(x)
    array = np.sort(x)
    array = array.astype(float)+0.0000001  # values cannot be 0
    n = len(array)
    index = np.arange(1, n+1) #index per array element
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient


def load_item_counts(ui_df, num_items):
    item_counts = Counter(ui_df['item'].values)
    for i in range(num_items):
        if i not in item_counts.keys():
            item_counts[i] = 0
    return item_counts
