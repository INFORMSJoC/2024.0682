import numpy as np
import torch

import utils
import sampler
import world


def BPR_train_original(dataset, recommend_model, bpr, epoch):
    Recmodel = recommend_model
    Recmodel.forecast = False
    Recmodel.train()
    S = sampler.Sample_interaction(dataset, world.ng_num)

    users = torch.Tensor(S[:, 0]).long().to(world.device)
    posItems = torch.Tensor(S[:, 1]).long().to(world.device)
    negItems = torch.Tensor(S[:, 2]).long().to(world.device)
    posTids = torch.Tensor(S[:, 3]).long().to(world.device)
    users, posItems, negItems, posTids = utils.shuffle(users, posItems, negItems, posTids)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg,
          batch_Tid)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   posTids,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stepOneBatch(batch_users, batch_pos, batch_neg, batch_Tid)
        aver_loss += cri
    bpr.stepOneEpoch()
    aver_loss = aver_loss / total_batch
    return aver_loss


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, hr, ndcg = [], [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        hr.append(utils.HR_ATk(groundTrue, r, k))
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'hr': np.array(hr),
            'ndcg': np.array(ndcg)}

def Test(dataset, Recmodel, cold=False, satisfication=False):
    u_batch_size = world.config['test_u_batch_size']
    if cold:
        testDict: dict = dataset.coldTestDict
    elif satisfication:
        testDict: dict = dataset.satisfactoryTestDict
    else:
        testDict: dict = dataset.testDict

    Recmodel.forecast = True
    item_counts = dataset.item_counts
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    results = {'hr': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // u_batch_size + 1
        # Recmodel.save_all_ratings()
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users, False)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x))
        for result in pre_results:
            results['hr'] += result['hr']
            results['ndcg'] += result['ndcg']
        results['hr'] /= float(len(users))
        results['ndcg'] /= float(len(users))

        for i, k in enumerate(world.topks):
            results[f'hr@{k}'] = results['hr'][i]
            results[f'ndcg@{k}'] = results['ndcg'][i]
        del results['hr'], results['ndcg']
        all_rating = torch.cat(rating_list, dim=0).cpu().numpy()
        for k in world.topks:
            ret = utils.diversity_at_k(all_rating, item_counts, dataset.niche_items, k)
            results[f'novelty@{k}'] = ret['novelty']
            results[f'niche_rate@{k}'] = ret['niche_rate']
        # print(results)
        return results


def Evaluate(dataset, Recmodel, epoch, cold=False, w=None):
    u_batch_size = world.config['test_u_batch_size']
    valDict: dict = dataset.valDict
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    item_counts = dataset.item_counts
    Recmodel.forecast = True
    results =  {
               'hr': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(valDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // u_batch_size + 1
        # Recmodel.save_all_ratings()
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [valDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x))
        for result in pre_results:
            results['hr'] += result['hr']
            results['ndcg'] += result['ndcg']
        results['hr'] /= float(len(users))
        results['ndcg'] /= float(len(users))

        for i, k in enumerate(world.topks):
            results[f'hr@{k}'] = results['hr'][i]
            results[f'ndcg@{k}'] = results['ndcg'][i]
        del results['hr'], results['ndcg']
        # del results['recall'], results['precision'], results['hr'], results['ndcg']
        all_rating = torch.cat(rating_list, dim=0).cpu().numpy()
        for k in world.topks:
            ret = utils.diversity_at_k(all_rating, item_counts, dataset.niche_items, k)
            results[f'novelty@{k}'] = ret['novelty']
            results[f'niche_rate@{k}'] = ret['niche_rate']
        return results
