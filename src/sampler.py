from time import time
import numpy as np
from torch import optim


class BPRLoss:
    def __init__(self, recmodel, config):
        self.model = recmodel
        self.iter_num = 0
        self.emb_l2rg = config['emb_l2rg']
        self.init_lr = config['lr']
        self.lr_gamma = 0.001
        self.decay_rate = 0.95
        self.opt = optim.Adam(recmodel.parameters(), lr=self.init_lr)

    def stepOneBatch(self, users, pos, neg, Tid):
        self.opt.zero_grad()
        loss, reg_loss = self.model.bpr_loss(users, pos, neg, Tid)
        reg_loss = reg_loss * self.emb_l2rg
        loss = loss + reg_loss
        loss.backward()
        self.opt.step()
        return loss.cpu().item()

    def get_lr(self) -> float:
        lr = self.init_lr * (1 + self.lr_gamma * self.iter_num) ** (-self.decay_rate)
        return lr

    def stepOneEpoch(self):
        lr = self.get_lr()
        # print(f'lr update to {lr}')
        self.iter_num += 1
        for param_group in self.opt.param_groups:
            if "lr_mult" not in param_group:
                param_group["lr_mult"] = 1
            param_group['lr'] = lr * param_group["lr_mult"]


def UniformSample_user(dataset):
    users = list(range(dataset.n_users))
    allPos = dataset.allPos
    S = []
    for i, user in enumerate(users):
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem, 0])
    return np.array(S)



# def UniformSample_ui(dataset):
#     interaction_num = dataset.trainDataSize
#     users = np.random.randint(0, dataset.n_users, interaction_num)
#     allPos = dataset.allPos
#     S = []
#     for i, user in enumerate(users):
#         posForUser = allPos[user]
#         if len(posForUser) == 0:
#             continue
#         posindex = np.random.randint(0, len(posForUser))
#         positem = posForUser[posindex]
#         while True:
#             negitem = np.random.randint(0, dataset.m_items)
#             if negitem in posForUser:
#                 continue
#             else:
#                 break
#         S.append([user, positem, negitem])
#     return np.array(S)


def Sample_interaction(dataset, num_ng):
    S = []
    allPos = dataset.allPos
    for user, item, timestamp in zip(dataset.trainUser, dataset.trainItem, dataset.trainTimestamp):
        for i in range(num_ng):
            neg_item = sample_neg_item(user, dataset.m_items, allPos[user])
            S.append([user, item, neg_item, timestamp])
    return np.array(S)


def find_popularity_array(items, timestamps, df):
    items_np = items.cpu().numpy()
    timestamps_np = timestamps.cpu().numpy()

    popularity_results = []
    for item, timestamp in zip(items_np, timestamps_np):
        item_data = df[df['item'] == item].sort_values('timestamp')

        if item_data.empty:
            popularity_results.append(0)  # 如果 item 不在数据中，返回 0
        else:
            popularity = np.interp(timestamp, item_data['timestamp'], item_data['popularity'])
            popularity_results.append(popularity)

    return np.array(popularity_results)


def sample_neg_item(user, item_num, exclude_item):
    neg_item = np.random.randint(low=0, high=item_num)
    while neg_item in exclude_item:
        neg_item = np.random.randint(low=0, high=item_num)
    return neg_item


def sample_from_hist(sample_arr, num_sample, masked_value):
    if len(sample_arr) == 0:
        return np.zeros((num_sample), dtype=int)+masked_value
    indices = np.arange(len(sample_arr))
    # exclude_index = np.where(sample_arr == input_item)[0]
    # indices = np.delete(indices, exclude_index)

    if len(indices) == 0:
        return np.zeros((num_sample,), dtype=int)
    if len(indices) > num_sample:
        idx = np.random.choice(indices, num_sample, replace=False)
        hist = sample_arr[idx]
    else:
        padding = np.zeros((num_sample - len(indices)), dtype=int)+masked_value
        hist = np.hstack([sample_arr[indices].flatten(), padding])
    return hist


def sample_from_nbr(sample_arr, num_sample, masked_value, input_user):
    sample_arr = np.array(sample_arr, dtype=int)
    if len(sample_arr) == 0:
        return np.zeros((num_sample), dtype=int) + masked_value
    indices = np.arange(len(sample_arr)).astype(int)
    if len(indices) >= num_sample:
        idx = np.random.choice(indices, num_sample, replace=False)
        pos_nbr = sample_arr[idx]
    else:
        nbrs = sample_arr[indices].flatten()
        nbrs = np.append(nbrs, input_user, axis=None)
        padding = np.zeros((num_sample - len(indices) - 1), dtype=int) + masked_value
        pos_nbr = np.hstack([nbrs, padding])
    return pos_nbr


def sample_from_hist_pop(sample_arr, num_sample, masked_value, input_item, item_pop_dict, pop_num):
    if len(sample_arr) == 0:
        return np.zeros((num_sample), dtype=int)+masked_value, \
               np.zeros((num_sample), dtype=int)+pop_num
    indices = np.arange(len(sample_arr))
    exclude_index = np.where(sample_arr == input_item)[0]
    indices = np.delete(indices, exclude_index)

    if len(indices) == 0:
        return np.zeros((num_sample), dtype=int)+masked_value, \
               np.zeros((num_sample), dtype=int)+pop_num
    if len(indices) > num_sample:
        idx = np.random.choice(indices, num_sample, replace=False)
        hist = sample_arr[idx]
    else:
        padding = np.zeros((num_sample - len(indices)), dtype=int)+masked_value
        hist = np.hstack([sample_arr[indices].flatten(), padding])
    hist_pop = np.array([item_pop_dict.get(i, pop_num) for i in hist])

    return hist, hist_pop


def sample_from_nbr_pop(sample_arr, num_sample, masked_value, user_degree_dict, degree_num):
    sample_arr = np.array(sample_arr, dtype=int)
    if len(sample_arr) == 0:
        return np.zeros((num_sample), dtype=int)+masked_value, \
               np.zeros((num_sample), dtype=int)+degree_num
    indices = np.arange(len(sample_arr)).astype(int)
    if len(indices) > num_sample:
        idx = np.random.choice(indices, num_sample, replace=False)
        pos_nbr = sample_arr[idx]
    else:
        padding = np.zeros((num_sample - len(indices)), dtype=int)+masked_value
        pos_nbr = np.hstack([sample_arr[indices].flatten(), padding])
    nbr_degree =  np.array([user_degree_dict.get(i, degree_num) for i in pos_nbr])
    return pos_nbr, nbr_degree
