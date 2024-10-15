import pickle
from collections import defaultdict
import numpy as np
import argparse
import os
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='Appliances')
    parser.add_argument("--aug_num", type=int, default=1) # for each user, add this number of interactions
    parser.add_argument("--strategy", type=str, default="random")
    parser.add_argument("--topK", type=int, default=5) # topk for most similar items
    return parser.parse_args()

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
if __name__ == "__main__":
    seed_everything(0)
    
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_prefix = './{}/preprocessed/'.format(args.dataset_name)

    # create directory if not exist
    aug_path = './{}/aug/'.format(args.dataset_name)
    if not os.path.exists(aug_path):
        os.mkdir(aug_path)
        print("path has been created: ", aug_path)
    args.aug_path = aug_path

    if 'sentence_transformer' in args.strategy:
        emb_filename = dataset_prefix + 'embs_{}.pth'.format(args.strategy)
        embs = torch.load(emb_filename)
        embs = torch.from_numpy(embs)
        data = embs.to(device)
        similarity_matrix = torch.mm(data, data.t())
        similarity_matrix[range(len(similarity_matrix)), range(len(similarity_matrix))] = torch.tensor([-float('inf')]*len(similarity_matrix)).to(device)
        
        # set popular item scores to low values so that encourage add interactions with unpopular items
        item_popularity_filename = dataset_prefix + 'item_popularity_label.pkl'
        with open(item_popularity_filename, 'rb') as f:
            pop, unpop = pickle.load(f)
        pop = list(pop)
        similarity_matrix[:, pop] = -float('inf')
            
        k_nearest_values, k_nearest_indices = torch.topk(similarity_matrix, args.topK, dim=1)
    elif args.strategy == 'random':
        title_filename = dataset_prefix + 'idx_to_title.pkl' # just use to obtain the item size
        with open(title_filename, 'rb') as f:
            item_titles = pickle.load(f)
        item_num = len(item_titles)
        k_nearest_indices = torch.randint(low=0, high=item_num-1, size=(item_num, args.topK))

    user_activeness_filename = dataset_prefix + 'user_activeness_label.pkl'
    with open(user_activeness_filename, 'rb') as f:
        active, inactive = pickle.load(f)

    query_users = inactive # augment for all inactive users
    
    edges_filename = dataset_prefix + 'edges.pkl'
    with open(edges_filename, 'rb') as f:
        train_edges, val_edges, test_edges = pickle.load(f)
    interaction_dict = defaultdict(list)
    for u, v in train_edges:
        if u in query_users:
            interaction_dict[u].append(v)

    aug_edges = []
    for u in query_users:
        interacted_items = interaction_dict[u]
        close_items = []
        for i in interacted_items:
            close_items.extend(k_nearest_indices[i].tolist())
        aug_item = np.random.choice(close_items, args.aug_num)
        for a in aug_item:
            aug_edges.append([u, a])

    # remove edges falling in val or test
    len_before = len(aug_edges)
    set_aug = set(map(tuple, aug_edges))
    set_val = set(map(tuple, val_edges))
    set_test = set(map(tuple, test_edges))
    set_aug = set_aug - set_val
    set_aug = set_aug - set_test
    aug_edges = list(map(list, set_aug))
    len_after = len(aug_edges)
    with open(aug_path+'aug_edges_{}_{}.pkl'.format(args.strategy, str(args.aug_num)), 'wb') as f:
        pickle.dump(aug_edges, f)
