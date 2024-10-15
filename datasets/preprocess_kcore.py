import json
import pickle
import gzip
from collections import defaultdict
import numpy as np
import argparse
import os
import pandas as pd
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='Appliances')
    parser.add_argument("--kcore_num", type=int, default=5)
    return parser.parse_args()

def non_repeated_items(args):
    meta_filename = './{}/src/meta_{}.jsonl.gz'.format(args.dataset_name, args.dataset_name)
    
    title_to_id = defaultdict(list)
    with gzip.open(meta_filename, 'rt') as file:
        for line in file:
            data = json.loads(line.strip())
            item_id = data['parent_asin']
            title = data['title']
            if title is None:
                continue
            title_to_id[title].append(item_id)
    
    number_of_items = np.array([len(v) for k, v in title_to_id.items()])
    print('Non-repeat Ratio:', round(np.sum(number_of_items == 1)/sum(number_of_items), 4))
    id_to_title = {v[0]: k for k, v in title_to_id.items() if len(v) == 1}
    return id_to_title

def load_edge_list(args, item_subset):
    edges = []
    dataset_name = args.dataset_name
    with gzip.open('./{}/src/{}.csv.gz'.format(dataset_name, dataset_name), 'rt') as file:
        test_df = pd.read_csv(file)
        for instance in test_df.itertuples():
            user_id = instance.user_id
            item_id = instance.parent_asin
            rating = instance.rating
            if rating <= 3 or item_id not in item_subset:
                continue
            edges.append([user_id, item_id])
    print('Number of edges:', str(len(edges)))
    return edges

def kcore(edges, kcore_num):
    src_degree = defaultdict(int)
    target_degree = defaultdict(int)

    for u, v in edges:
        src_degree[u] += 1
        target_degree[v] += 1

    src_list = [u for u, d in src_degree.items() if d >= kcore_num]
    target_list = [u for u, d in target_degree.items() if d >= kcore_num]
    
    new_edges = []
    src_set = set(src_list)
    target_set = set(target_list)
    for u, v in edges:
        if u in src_set and v in target_set:
            new_edges.append([u, v])
    return new_edges

def categorize_items(ratings_dict): 
    # input is a dict with item as key and the number of interactions as value
    total_ratings = sum(ratings_dict.values())    
    sorted_items = sorted(ratings_dict.items(), key=lambda x: x[1], reverse=True)    
    cumulative_ratings = 0
    threshold = 0.8 * total_ratings  # 80% of total ratings
    short_head = []
    long_tail = []
    for item_id, rating_count in sorted_items:
        cumulative_ratings += rating_count
        if cumulative_ratings <= threshold:
            short_head.append(item_id)
        else:
            long_tail.append(item_id)
    return set(short_head), set(long_tail)

def group_label_assignment(train_edges):
    # Group Label Assignment: assign user labels of active/inactive and item labels of popular and unpopular
    item_interaction_dict = defaultdict(int)
    user_interaction_dict = defaultdict(int)
    for u, v in train_edges:
        item_interaction_dict[v] += 1
        user_interaction_dict[u] += 1
    print('Item average interactions (num, avg_degree, min, max):', len(item_interaction_dict), round(np.mean(list(item_interaction_dict.values())), 2), min(list(item_interaction_dict.values())), max(list(item_interaction_dict.values())))
    print('User average interactions (num, avg_degree, min, max):', len(user_interaction_dict), round(np.mean(list(user_interaction_dict.values())), 2), min(list(user_interaction_dict.values())), max(list(user_interaction_dict.values())))
    
    # obtain the labels for items
    short_head, long_tail = categorize_items(item_interaction_dict)
    print("Short Head Items (num, avg_degree, min, max):", len(short_head), round(np.mean([item_interaction_dict[k] for k in short_head]), 2), np.min([item_interaction_dict[k] for k in short_head]), np.max([item_interaction_dict[k] for k in short_head]))
    print("Long Tail Items (num, avg_degree, min, max):", len(long_tail), round(np.mean([item_interaction_dict[k] for k in long_tail]), 2), np.min([item_interaction_dict[k] for k in long_tail]), np.max([item_interaction_dict[k] for k in long_tail]))

    # obtain the labels for users
    users_list = list(user_interaction_dict.items())
    users_list.sort(key=lambda x: x[1], reverse=True)
    top_20_percent_count = int(len(users_list) * 0.2)
    active = users_list[:top_20_percent_count]
    active = set([t[0] for t in active])
    inactive = set([t[0] for t in users_list]) - active

    print('Active users (num, avg_degree, min, max):', len(active), round(np.mean([user_interaction_dict[k] for k in active]),2), np.min([user_interaction_dict[k] for k in active]), np.max([user_interaction_dict[k] for k in active]))
    print('Inactive users (num, avg_degree, min, max):', len(inactive), round(np.mean([user_interaction_dict[k] for k in inactive]), 2), np.min([user_interaction_dict[k] for k in inactive]), np.max([user_interaction_dict[k] for k in inactive]))

    return short_head, long_tail, active, inactive

def save_pickle(filename, obj):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)


if __name__ == "__main__":
    args = parse_args()
    random.seed(0)

    print('----------------' + args.dataset_name + '------------------')

    # create directory if not exist
    preprocess_path = './{}/preprocessed/'.format(args.dataset_name)
    if not os.path.exists(preprocess_path):
        os.mkdir(preprocess_path)
        print("path has been created: ", preprocess_path)
    args.preprocess_path = preprocess_path

    # preprocess the meta data to keep the items with unique titles
    id_to_title = non_repeated_items(args)
    
    # obtain edges within the item set and have positive ratings
    edges = load_edge_list(args, set(id_to_title.keys()))
    
    # perform kcore iteratively until convergence
    last_edge_num = 0
    while last_edge_num != len(edges):
        # print(last_edge_num, len(edges))
        last_edge_num = len(edges)
        edges = kcore(edges, args.kcore_num)
    print('Number of edges after kcore:', len(edges))

    total_users = set([u for u, v in edges])
    total_items = set([v for u, v in edges])
    print('Number of users/items:', len(total_users), len(total_items))

    user_id_to_idx = {k: i for i, k in enumerate(list(total_users))}
    item_id_to_idx = {k: i for i, k in enumerate(list(total_items))}

    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)

    # split into train/val/test based on 60%/20%/20%
    train_edges, val_edges, test_edges = [], [], []
    for u, vs in adj.items():
        random.shuffle(vs)
        train_cnt = int(len(vs)*0.6)
        val_cnt = int(len(vs)*0.2)
        test_cnt = len(vs)-train_cnt-val_cnt
        if test_cnt == 0 or val_cnt == 0:
            test_cnt = 1
            val_cnt = 1
            train_cnt = len(vs)-test_cnt-val_cnt
        u = user_id_to_idx[u]
        vs = [item_id_to_idx[v] for v in vs]
        train_edges.extend([[u, v] for v in vs[:train_cnt]])
        val_edges.extend([[u, v] for v in vs[train_cnt: train_cnt+val_cnt]])
        test_edges.extend([[u, v] for v in vs[-test_cnt:]])
    for e in [train_edges, val_edges, test_edges]:
        print('User number:', len(set([u for u, v in e])), 'Item number:', len(set([v for u, v in e])))
    
    idx_to_title = dict()
    for k, v in id_to_title.items():
        if k in item_id_to_idx:
            idx = item_id_to_idx[k]
            idx_to_title[idx] = v

    short_head, long_tail, active, inactive = group_label_assignment(train_edges)

    # saving documents
    save_pickle(preprocess_path + 'edges.pkl', [train_edges, val_edges, test_edges])
    print('Finish saving train/val/test edges to file:', len(train_edges), len(val_edges), len(test_edges))
    save_pickle(preprocess_path + 'id_to_idx.pkl',[user_id_to_idx, item_id_to_idx])
    print('Finish saving index to file (user num, item num):', len(user_id_to_idx), len(item_id_to_idx))
    save_pickle(preprocess_path + 'idx_to_title.pkl', idx_to_title)
    print('Finish saving idx_to_title:', len(idx_to_title))
    save_pickle(preprocess_path + 'item_popularity_label.pkl', (short_head, long_tail))
    save_pickle(preprocess_path + 'user_activeness_label.pkl', (active, inactive))
    print('Finish saving labels\n\n')
