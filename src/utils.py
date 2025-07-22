import os
import joblib
import torch
from sklearn.metrics import roc_auc_score


import world



def save_pkl(file, obj, compress=0):
    # compress=('gzip', 3)
    joblib.dump(value=obj, filename=file, compress=compress)


def load_pkl(file):
    return joblib.load(file)



def set_seed(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def getFileName(model_name):
    file = f"{model_name}-{world.config['latent_dim_rec']}-{world.config['layer']}layer-{world.config['social_layer']}social_layer"

    return file


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def HR_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    score = np.count_nonzero(right_pred)
    return score

def diversity_at_k(pred_data, item_popularity, niche_items, k):
    sum_novelty = []
    sum_niche_rates = []
    u_num = len(pred_data)
    for u in range(u_num):
        predictTopK = pred_data[u][:k]
        novelty = 0.0
        niche_num = 0.0
        for i in predictTopK:
            try:
                novelty += (np.log2(u_num / item_popularity[int(i)]))
                if i in niche_items:
                    niche_num+=1
            except:
                novelty += (np.log2(u_num / 1))
                if i in niche_items:
                    niche_num+=1
        novelty = novelty / k
        niche_rate = niche_num/ k
        sum_novelty.append(novelty)
        sum_niche_rates.append(niche_rate)

    novelty = np.average(sum_novelty)
    niche_rate = np.average(sum_niche_rates)*100

    return {'novelty': novelty,  "niche_rate":niche_rate}

def MRRatK_r(r, k):
    pred_data = r[:, :k]
    scores = np.log2(1. / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def AUC(all_item_scores, dataset, test_data):
    r_all = np.zeros((dataset.m_items,))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

# ====================end Metrics=============================
# =========================================================
import numpy as np

def calculate_metrics(df):
    # Number of unique users
    num_users = df['user'].nunique()

    # Number of unique items
    num_items = df['item'].nunique()

    # Total number of ratings
    num_ratings = len(df)

    # Sparsity
    sparsity = (1 - num_ratings / (num_users * num_items)) * 100

    # Average number of ratings per item
    avg_ratings_per_item = num_ratings / num_items

    # Calculate the distribution of ratings per item
    item_ratings_count = df['item'].value_counts()

    niche_items = item_ratings_count[item_ratings_count<avg_ratings_per_item].index.values
    niche_items = set(niche_items)

    # Percentage of ratings from top 10% items
    top_10_percent_count = int(num_items * 0.1)
    top_10_percent_ratings = item_ratings_count[:top_10_percent_count].sum()
    percent_ratings_top_10 = (top_10_percent_ratings / num_ratings) * 100

    # Percentage of ratings from top 20% items
    top_20_percent_count = int(num_items * 0.2)
    top_20_percent_ratings = item_ratings_count[:top_20_percent_count].sum()
    percent_ratings_top_20 = (top_20_percent_ratings / num_ratings) * 100

    # Percentage of ratings from top 50% items
    top_50_percent_count = int(num_items * 0.5)
    top_50_percent_ratings = item_ratings_count[:top_50_percent_count].sum()
    percent_ratings_top_50 = (top_50_percent_ratings / num_ratings) * 100

    # Gini coefficient
    def gini(array):
        array = np.sort(array) # Sort array
        array = array.astype(float) + 0.0000001  # values cannot be 0
        n = len(array)
        index = np.arange(1, n + 1)  # index per array element
        return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

    gini_coefficient = gini(item_ratings_count.values)

    # Print results
    print(f"#Users: {num_users}")
    print(f"#Items: {num_items}")
    print(f"#Ratings: {num_ratings}")
    print(f"Sparsity (%): {sparsity:.2f}")
    print(f"Avg. #Ratings per Item: {avg_ratings_per_item:.2f}")
    print(f"% of Ratings from Top 10% Items: {percent_ratings_top_10:.2f}")
    print(f"% of Ratings from Top 20% Items: {percent_ratings_top_20:.2f}")
    print(f"% of Ratings from Top 50% Items: {percent_ratings_top_50:.2f}")
    print(f"Gini Coefficient: {gini_coefficient:.4f}")

    return niche_items

# Example usage:
# calculate_metrics(df)
