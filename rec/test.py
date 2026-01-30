"""Evaluation entry point and fairness metrics."""

import os
from parse import parse_args
import time
import numpy as np
import pickle
from utils import *
from evaluation import *
from model import *
from dataprocess import *

def cal_user_unfairness(args):
    """Compute unfairness between active and inactive users.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        tuple: (active_avg, inactive_avg)
    """
    result_filename = args.per_user_filename
    with open(result_filename, 'rb') as f:
        user_result = pickle.load(f)
    activeness_filename = "../datasets/{}/preprocessed/user_activeness_label.pkl".format(args.dataset)
    with open(activeness_filename, 'rb') as f:
        active, inactive = pickle.load(f)

    active_performance = []
    for u in active:
        p = np.mean(user_result[u][0]) # 5 metrics, only one record for each user
        active_performance.append(p)
    active_avg = np.mean(active_performance)

    inactive_performance = []
    for u in inactive:
        p = np.mean(user_result[u][0])
        inactive_performance.append(p)
    inactive_avg = np.mean(inactive_performance)
    
    return active_avg, inactive_avg

def cal_item_unfairness(args):
    """Compute unfairness between popular and unpopular items.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        tuple: (pop_avg, unpop_avg)
    """
    result_filename = args.per_item_filename
    with open(result_filename, 'rb') as f:
        item_results = pickle.load(f)
    popularity_file = "../datasets/{}/preprocessed/item_popularity_label.pkl".format(args.dataset)
    with open(popularity_file, 'rb') as f:
        pop, unpop = pickle.load(f)
    
    pop_performance = []
    unpop_performance = []
    for i in item_results.keys():
        result = item_results[i]
        if i in pop:
            pop_performance.extend([np.mean(p) for p in result])
        else:
            unpop_performance.extend([np.mean(p) for p in result])
    pop_avg = np.mean(pop_performance)
    unpop_avg = np.mean(unpop_performance)
    return pop_avg, unpop_avg
    
if __name__ == '__main__':
    args = parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args.device)
    args.path = os.getcwd()
    args.dataset_path = '../datasets/{}/'.format(args.dataset)

    seed_everything(args.seed)
    print("Configurations:", args)

    """build dataset"""
    train_cf, val_cf, test_cf, user_dict, args.n_users, args.n_items, clicked_set, adj = load_data(args)

    print(args.n_users, args.n_items, train_cf.shape[0] + val_cf.shape[0], test_cf.shape[0],
        (train_cf.shape[0] + val_cf.shape[0] + test_cf.shape[0]) / (args.n_items * args.n_users))

    if(args.neg_in_val_test == 1):  # whether negative samples from validation and test sets
        clicked_set = user_dict['train_user_set']

    """build model"""
    if args.model == 'LightGCN':
        model = LightGCN(args).to(args.device)
    elif args.model == 'LightGCN_with_content':
        model = LightGCN_with_content(args).to(args.device)
    elif args.model == 'LightGCN_with_content_id_add':
        model = LightGCN_with_content_id_add(args).to(args.device)
    elif args.model == 'LightGCN_with_content_id_cat':
        model = LightGCN_with_content_id_cat(args).to(args.device)
    elif args.model == 'MF':
        model = MF(args).to(args.device)

    adj_sp_norm, deg = normalize_edge(adj, args.n_users, args.n_items)
    edge_index, edge_weight = adj_sp_norm._indices(), adj_sp_norm._values()

    model.adj_sp_norm = adj_sp_norm.to(args.device)
    model.edge_index = edge_index.to(args.device)
    model.edge_weight = edge_weight.to(args.device)
    model.deg = deg.to(args.device)

    if args.flag == 0:
        args.save_model_filename = './trained_models/{}/{}_{}.pkl'.format(args.dataset, args.model, str(args.seed))
    elif args.flag == 1:
        args.save_model_filename = './trained_models/{}/aug_{}_{}_{}_{}.pkl'.format(args.dataset, args.model, args.strategy, str(args.aug_num), str(args.seed))
    elif args.flag == 2:
        args.save_model_filename = './trained_models/{}/aug_user_{}_{}_{}_{}.pkl'.format(args.dataset, args.model, args.strategy, str(args.aug_num), str(args.seed))
    model.load_state_dict(torch.load(args.save_model_filename))
    model.eval()

    # evaluate and save the results
    sparsity = (train_cf.shape[0] + val_cf.shape[0] + test_cf.shape[0]) / (args.n_items * args.n_users)
    results = [args.seed, sparsity]
    if not os.path.exists('./results_logs'):
        os.mkdir('./results_logs')
    if args.flag == 0:
        save_results_filename = './results_logs/{}_{}.txt'.format(args.dataset, args.model)
    elif args.flag == 1:
        save_results_filename = './results_logs/{}_{}_{}_{}.txt'.format(args.dataset, args.model, args.strategy, str(args.aug_num))
    elif args.flag == 2:
        save_results_filename = './results_logs/user_{}_{}_{}_{}.txt'.format(args.dataset, args.model, args.strategy, str(args.aug_num))
    # utility performance
    time_start = time.time()
    user_embs, item_embs = model.generate()
    test_res = test(user_embs, item_embs, user_dict, args, flag='test')
    print('eval time:', time.time()-time_start)
    results.extend([test_res['Recall'][0], test_res['NDCG'][0], test_res['Precision'][0], test_res['Hit_ratio'][0], test_res['F1'][0]])
    
    utility_measurement = (test_res['Recall'][0]+test_res['NDCG'][0]+test_res['Precision'][0]+test_res['Hit_ratio'][0]+test_res['F1'][0])/5
    print('Recall:', test_res['Recall'], 'NDCG:', test_res['NDCG'])
    print('Avg Utility:', round(utility_measurement, 4))
    print(round(test_res['Recall'][0], 4), round(test_res['NDCG'][0], 4), round(utility_measurement, 4), round((train_cf.shape[0] + val_cf.shape[0] + test_cf.shape[0]) / (args.n_items * args.n_users), 6))

    # fairness performance
    test_per_user(user_embs, item_embs, user_dict, args)
    test_per_item(user_embs, item_embs, user_dict, args)
    pop_avg, unpop_avg = cal_item_unfairness(args)
    active_avg, inactive_avg = cal_user_unfairness(args)
    results.extend([pop_avg, unpop_avg, active_avg, inactive_avg])

    results_str = " ".join([str(t) for t in results])
    results_str += "\n"
    with open(save_results_filename, 'a+') as f:
        f.write(results_str)
        
