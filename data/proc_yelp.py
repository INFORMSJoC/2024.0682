#! /bin/usr/python
import sys
sys.path.append(".")
import os
import argparse
from collections import Counter

import pandas as pd
import numpy as np
from tqdm import tqdm

try:
    import ujson as json
except ImportError:
    import json

from src.utils import load_pkl, save_pkl


# 定义根路径和数据路径
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT_PATH, 'data', 'yelp')
PARSE_ROOT_DIR = os.path.join(ROOT_PATH, 'data')
PARSE_YELP_DIR = os.path.join(ROOT_PATH, 'data', 'yelp')
PREPROCESS_DIR = os.path.join(ROOT_PATH, 'data', 'yelp')
# CITY_DIR = "./data/parse/yelp/citycluster/"
# INTERACTION_DIR = "./data/parse/yelp/interactions/"
# TRAIN_TEST_DIR = "./data/parse/yelp/train_test/"
# 'Philadelphia', 'Tucson', 'Tampa'
# CITY_NAME_ABBR = {"Philadelphia": "phi"}
CANDIDATE_CITY = ['Tucson']



def load_user_business():
    """helper function that load user and business"""
    user_profile = load_pkl(os.path.join(PREPROCESS_DIR, "user_profile.pkl"))
    business_profile = load_pkl(os.path.join(PREPROCESS_DIR, "business_profile.pkl"))
    return user_profile, business_profile


def parse_user():
    """Load users
    output: id2user.pkl,
            user2id.pkl,
            user.friend.pkl,
            user.profile.pkl
    """
    user_profile = {}
    user_friend = {}

    print("\t[parse user] load user list")
    users_list = load_pkl(os.path.join(PREPROCESS_DIR, "users_list.pkl"))
    users_list = set(users_list)

    print("\t[parse user] building user profiles")
    with open(DATA_DIR + "yelp_academic_dataset_user.json", "r",encoding='utf-8') as fin:
        for ind, ln in enumerate(fin):
            data = json.loads(ln)
            user_id = data['user_id']
            if user_id not in users_list:  # discard infrequent or irrelevant cities
                continue
            user_friend[user_id] = data['friends'].split(", ")
            del data['friends']
            del data['user_id']
            user_profile[user_id] = data

    # user adjacency and profile dictionary separately
    print("\t[parse user] dumping user-friendship and user-profile information ...")
    save_pkl(os.path.join(PREPROCESS_DIR, "user_friend.pkl"), user_friend)
    save_pkl(os.path.join(PREPROCESS_DIR, "user_profile.pkl"), user_profile)


def parse_business():
    """extract business information from business.json
    output:
        business.profile.pkl
        city.business.pkl
    """

    city_business = {}  # dictionary of city: [business list]
    business_profiles = {}  # dictionary of business profile

    # count business by location (city and state)
    print("\t[parse_business] preprocessing all business without selecting cities ...")
    with open(DATA_DIR + "yelp_academic_dataset_business.json", "r", encoding='utf-8') as fin:
        for ind, ln in enumerate(fin):
            data = json.loads(ln)
            city = data['city']

            if city not in CANDIDATE_CITY:  # only use cities
                continue

            business_id = data["business_id"]
            # removed fields: id, state, attributes, and hours
            # remained fields: fields: name, address, postal-code, latitude/longitude
            #              star, review_count, is_open
            del data["business_id"], data["state"]
            del data["attributes"], data["hours"]
            business_profiles[business_id] = data

            # save business id to city_business dictionary
            city_business[city] = city_business.get(city, [])
            city_business[city].append(business_id)

    # save city business mapping
    print("\t[parse business] dumping business.profile and city.business ...")
    save_pkl(os.path.join(PREPROCESS_DIR, "business_profile.pkl"), business_profiles)
    save_pkl(os.path.join(PREPROCESS_DIR, "city_business.pkl"), city_business)


def parse_interactions():
    """draw interact from `review.json` and `tips.json`.
    output: ub.interact.csv
    Args:
        keep_city - the interact of cities to keep
    """

    # business_profile only contains city in Lv, Tor, and Phx
    print("\t[parse interactions] loading business_profile pickle...")
    business_profile = load_pkl(os.path.join(PREPROCESS_DIR, "business_profile.pkl"))

    users, businesses, cities, stars = [], [], [], []
    timestamps = []

    # create records as (user, business, city) tuple
    print("\t[parse interactions] loading review.json ...")
    with open(DATA_DIR + "yelp_academic_dataset_review.json", "r", encoding='utf-8') as fin:
        for ln in fin:
            data = json.loads(ln)
            _bid = data['business_id']
            if _bid not in business_profile:  # only Lv, Tor, and Phx businesses
                continue
            users.append(data['user_id'])
            businesses.append(_bid)
            cities.append(business_profile[_bid]["city"])
            stars.append(data['stars'])
            timestamps.append(data['date'])

    interactions = pd.DataFrame({
        'user': users, 'business': businesses, "city": cities, "stars":stars,
        "timestamp": timestamps})

    interactions.to_csv(os.path.join(PREPROCESS_DIR, "user_business_interact.csv"), index=False)

    # kept user for parse user
    user_remained = interactions["user"].unique().tolist()
    save_pkl(os.path.join(PREPROCESS_DIR, "users_list.pkl"), user_remained)


def city_clustering(city, user_min_count, business_min_count,
        user_profile, business_profile, interactions, user_friendships):
    """city cluster create city-specific data,
    that is, narrow down information to specific cities.
    User and business ids are replaced to new ones.
    Friendships are filtered to users only in the same city.
    Args:
        city - city 
        user_min_count - threshold for users to be added into experiments.
        business_min_count - threshold for businesses to be add into experiments.
        user_profile - all user profiles.
        business_profile - all business profiles.
        interactions - all interactions among all cities.
        user_friendship - the users' friendship relations.
    Return:
        business_of_city: list
        user_of_city: list
        city_b2i, city_u2i: dict, reverse relationship
        city_user_frienship: new id, friendship
        city_user_profile: new id-profile
        city_business_profile: new id-profile
        interaction_of_city: csv files with new ids
    """
    print("\t[city_cluster] Processing city: {}".format(city))

    # make specific folder for city
    # city_dir = CITY_NAME_ABBR[city] + "/"
    # if not os.path.isdir(city_dir):
    #     os.mkdir(city_dir)

    city_user_friendship = {}  # new_id: friends in new_id
    city_user_profile = {}
    city_business_profile = {}

    interactions_of_city = interactions[interactions["city"] == city]

    # remove rear businesses and users appear less than min-count times
    print("\t\t[city_cluster] removing entries under min_count b:{}, u:{}"
          .format(business_min_count, user_min_count))
    b_counter = Counter(interactions_of_city.business)
    u_counter = Counter(interactions_of_city.user)

    interactions_of_city = interactions_of_city.assign(
        b_count=lambda x: x.business.map(b_counter))
    interactions_of_city = interactions_of_city.assign(
        u_count=lambda x: x.user.map(u_counter))
    interactions_of_city = interactions_of_city[
        (interactions_of_city.b_count >= business_min_count) &
        (interactions_of_city.u_count >= user_min_count)]

    user_of_city = interactions_of_city['user'].unique().tolist()  # list
    business_of_city = interactions_of_city["business"].unique().tolist()
    print("\t\t[city_cluster] # of users {}, # of business {}".format(
        len(user_of_city), len(business_of_city)))

    # ** before this point: old user/business id; 
    # ** after this point: new user/business index

    # user, business index starting from 1 to len(user_of_city)
    city_uid2ind = dict(zip(user_of_city, range(1, len(user_of_city) + 1)))
    city_bid2ind = dict(zip(business_of_city, range(1, len(business_of_city) + 1)))

    # create city friendships that are in the same city: city_user_friendship
    for uid in user_of_city:
        intersection = np.intersect1d(
            user_of_city, user_friendships[uid], assume_unique=True).tolist()
        city_user_friendship[city_uid2ind[uid]] = [city_uid2ind[x] for x in intersection]

    # create city specific user profile using new index: city_user_profile
    for uid in user_of_city:
        profile = user_profile[uid]
        profile["user_index"] = city_uid2ind[uid]
        city_user_profile[city_uid2ind[uid]] = profile

    # create city specific business profile using new index: city_business_profile
    for bid in business_of_city:
        profile = business_profile[bid]
        profile["business_id"] = city_bid2ind[bid]
        city_business_profile[city_bid2ind[bid]] = profile

    # user/business id to index in interactions
    interactions_of_city['user'] = interactions_of_city['user'].apply(
        lambda x: city_uid2ind[x])
    interactions_of_city['business'] = interactions_of_city['business'].apply(
        lambda x: city_bid2ind[x])

if __name__ == "__main__":
    city_name = 'Tucson'

    print("[--preprocess] parsing businesses/interactions/users from scratch ...")
    parse_business()
    parse_interactions()
    parse_user()
    print("[--preprocess] done!")

    print("[--city_cluster] running city cluster")
    print("\t[loading] processed files after preprocessing")
    user_profile = load_pkl(os.path.join(PREPROCESS_DIR, "user_profile.pkl"))
    business_profile = load_pkl(os.path.join(PREPROCESS_DIR, "business_profile.pkl"))
    city_business = load_pkl(os.path.join(PREPROCESS_DIR, "city_business.pkl"))
    interactions = pd.read_csv(os.path.join(PREPROCESS_DIR, "user_business_interact.csv"))
    user_friendships = load_pkl(os.path.join(PREPROCESS_DIR, "user_friend.pkl"))

    city_user_friendship = {}  # new_id: friends in new_id
    city_user_profile = {}
    city_business_profile = {}

    interactions_of_city = interactions[interactions["city"] == city_name]
    # remove rear businesses and users appear less than min-count times
    print("\t\t[city_cluster] removing entries under min_count b:{}, u:{}"
          .format(3, 3))
    b_counter = Counter(interactions_of_city.business)
    u_counter = Counter(interactions_of_city.user)

    interactions_of_city = interactions_of_city.assign(
        b_count=lambda x: x.business.map(b_counter))
    interactions_of_city = interactions_of_city.assign(
        u_count=lambda x: x.user.map(u_counter))
    interactions_of_city = interactions_of_city[
        (interactions_of_city.b_count >= 3) &
        (interactions_of_city.u_count >= 3)]

    user_of_city = interactions_of_city['user'].unique().tolist()  # list
    business_of_city = interactions_of_city["business"].unique().tolist()
    print("\t\t[city_cluster] # of users {}, # of business {}".format(
        len(user_of_city), len(business_of_city)))

    user_of_city = interactions_of_city['user'].unique().tolist()  # list
    business_of_city = interactions_of_city["business"].unique().tolist()
    print("\t\t[city_cluster] # of users {}, # of business {}".format(
        len(user_of_city), len(business_of_city)))

    # ** before this point: old user/business id;
    # ** after this point: new user/business index

    # user, business index starting from 1 to len(user_of_city)
    city_uid2ind = dict(zip(user_of_city, range(1, len(user_of_city) + 1)))
    city_bid2ind = dict(zip(business_of_city, range(1, len(business_of_city) + 1)))

    # create city friendships that are in the same city: city_user_friendship
    for uid in user_of_city:
        intersection = np.intersect1d(
            user_of_city, user_friendships[uid], assume_unique=True).tolist()
        city_user_friendship[city_uid2ind[uid]] = [city_uid2ind[x] for x in intersection]


    # create city specific business profile using new index: city_business_profile
    for bid in business_of_city:
        profile = business_profile[bid]
        profile["business_id"] = city_bid2ind[bid]
        if profile['categories'] != None:
            city_business_profile[city_bid2ind[bid]] = profile['categories'].split(', ')

    # user/business id to index in interactions
    interactions_of_city['user'] = interactions_of_city['user'].apply(
        lambda x: city_uid2ind[x])
    interactions_of_city['business'] = interactions_of_city['business'].apply(
        lambda x: city_bid2ind[x])

    df = interactions_of_city[['user','business','stars','timestamp']].rename(columns={'business':'item','stars':'rate'})
    uu_df = pd.DataFrame([(k,e) for k,v in city_user_friendship.items() for e in v],columns=['user','trust'])


    cate_df = pd.DataFrame([(k, e) for k, v in city_business_profile.items() for e in v], columns=['item', 'cate'])
    cate_key, cate_count = np.unique(cate_df['cate'], return_counts=True)
    cate_dict = dict(zip(cate_key, cate_count))
    cate_df = cate_df.assign(pop=lambda x: x.cate.map(cate_dict))
    cate_df = cate_df.sort_values('pop', ascending=False).groupby('item', as_index=False).first()
    item_cate = dict(cate_df[['item', 'cate']].values)
    df = df.assign(cate=lambda x: x.item.map(item_cate))

    df.to_csv( os.path.join(ROOT_PATH, 'data', city_name, 'ratings.csv'),index=False)
    uu_df.to_csv(os.path.join(ROOT_PATH, 'data', city_name, 'trust.csv'),index=False)