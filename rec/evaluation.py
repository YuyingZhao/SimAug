import numpy as np
from utils import *
from collections import defaultdict, Counter
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle

np.set_printoptions(precision=4)

def getLabel(test_data, pred_data):
    r = []

    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)

    return np.array(r).astype('float')


def Hit_at_k(r, k):
    right_pred = r[:, :k].sum(axis=1)

    return 1. * (right_pred > 0)


def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = right_pred / recall_n
    precis = right_pred / precis_n
    return {'Recall': recall, 'Precision': precis}


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """

    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix

    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = np.sum(pred_data * (1. / np.log2(np.arange(2, k + 2))), axis=1)

    idcg[idcg == 0.] = 1.  # it is OK since when idcg == 0, dcg == 0
    ndcg = dcg / idcg

    return ndcg


def test_one_batch(X, topks):
    sorted_items = X[0].numpy()
    groundTrue = X[1]

    r = getLabel(groundTrue, sorted_items)

    pre, recall, ndcg, hit_ratio, F1 = [], [], [], [], []
    for k in topks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        ndcgs = NDCGatK_r(groundTrue, r, k)
        hit_ratios = Hit_at_k(r, k)

        hit_ratio.append(sum(hit_ratios))
        pre.append(sum(ret['Precision']))
        recall.append(sum(ret['Recall']))
        ndcg.append(sum(ndcgs))

        temp = ret['Precision'] + ret['Recall']
        temp[temp == 0] = float('inf')
        F1s = 2 * ret['Precision'] * ret['Recall'] / temp
        # F1s[np.isnan(F1s)] = 0

        F1.append(sum(F1s))

    return {'Recall': np.array(recall),
            'Precision': np.array(pre),
            'NDCG': np.array(ndcg),
            'F1': np.array(F1),
            'Hit_ratio': np.array(hit_ratio)}


def test(user_embs, item_embs, user_dict, args, flag='val', model=None):
    results = {'Precision': np.zeros(len(args.topks)),
               'Recall': np.zeros(len(args.topks)),
               'NDCG': np.zeros(len(args.topks)),
               'Hit_ratio': np.zeros(len(args.topks)),
               'F1': np.zeros(len(args.topks))}

    train_user_set = user_dict['train_user_set']
    val_user_set = user_dict['val_user_set']

    if flag == "test":
        test_user_set = user_dict['test_user_set']
        test_users = torch.tensor(list(test_user_set.keys()))
    elif flag == "val":
        test_user_set = user_dict['val_user_set']
        test_users = torch.tensor(list(test_user_set.keys()))

    with torch.no_grad():
        users_list = []
        ratings_list = []
        groundTruth_items_list = []

        for batch_users in minibatch(test_users, batch_size=args.test_batch_size):
            batch_users = batch_users.to(args.device)
            rating_batch = torch.matmul(user_embs[batch_users], item_embs.t())

            # to introduce novelty of the recommended items
            # not recommend clicked items
            if flag == "val":
                clicked_items = [train_user_set[user.item()] - args.n_users
                                 for user in batch_users]
            elif flag == "test":
                clicked_items = []
                for user in batch_users:
                    clicked_items_for_user_in_train = train_user_set[user.item()] - args.n_users
                    clicked_items_for_user_in_val = val_user_set[user.item()] - args.n_users
                    clicked_items.append(np.concatenate([clicked_items_for_user_in_train, clicked_items_for_user_in_val]))

            groundTruth_items = [test_user_set[user.item()] - args.n_users
                                 for user in batch_users]

            exclude_index = []
            exclude_items = []

            for range_i, items in enumerate(clicked_items):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)

            rating_batch[exclude_index, exclude_items] = -(1 << 10)

            rating_K = torch.topk(rating_batch, k=max(args.topks))[1].cpu()

            users_list.append(batch_users)
            ratings_list.append(rating_K)
            groundTruth_items_list.append(groundTruth_items)

        X = zip(ratings_list, groundTruth_items_list)

        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x, args.topks))

        for result in pre_results:
            results['Recall'] += result['Recall']
            results['Precision'] += result['Precision']
            results['NDCG'] += result['NDCG']
            results['F1'] += result['F1']
            results['Hit_ratio'] += result['Hit_ratio']

        results['Recall'] /= len(test_users)
        results['Precision'] /= len(test_users)
        results['NDCG'] /= len(test_users)
        results['F1'] /= len(test_users)
        results['Hit_ratio'] /= len(test_users)

    return results

def test_rec_list(user_embs, item_embs, user_dict, args, flag='val', model=None):
    train_user_set = user_dict['train_user_set']
    val_user_set = user_dict['val_user_set']

    if flag == "test":
        test_user_set = user_dict['test_user_set']
        test_users = torch.tensor(list(test_user_set.keys()))
    elif flag == "val":
        test_user_set = user_dict['val_user_set']
        test_users = torch.tensor(list(test_user_set.keys()))

    with torch.no_grad():
        users_list = []
        ratings_list = []
      
        for batch_users in minibatch(test_users, batch_size=args.test_batch_size):
            batch_users = batch_users.to(args.device)
            if args.model == 'UltraGCN':
                rating_batch = model.predict(batch_users)
            else:
                rating_batch = torch.matmul(user_embs[batch_users], item_embs.t())

            # to introduce novelty of the recommended items
            # not recommend clicked items
            if flag == "val":
                clicked_items = [train_user_set[user.item()] - args.n_users
                                 for user in batch_users]
            elif flag == "test":
                clicked_items = []
                for user in batch_users:
                    clicked_items_for_user_in_train = train_user_set[user.item()] - args.n_users
                    clicked_items_for_user_in_val = val_user_set[user.item()] - args.n_users
                    clicked_items.append(np.concatenate([clicked_items_for_user_in_train, clicked_items_for_user_in_val]))

            groundTruth_items = [test_user_set[user.item()] - args.n_users
                                 for user in batch_users]

            exclude_index = []
            exclude_items = []

            for range_i, items in enumerate(clicked_items):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)

            rating_batch[exclude_index, exclude_items] = -(1 << 10)

            rating_K = torch.topk(rating_batch, k=max(args.topks))[1].cpu()

            users_list.append(batch_users)
            ratings_list.append(rating_K)
    return users_list, ratings_list

def test_per_user(user_embs, item_embs, user_dict, args):
    train_user_set = user_dict['train_user_set']
    val_user_set = user_dict['val_user_set']
    test_user_set = user_dict['test_user_set']
    test_users = torch.tensor(list(test_user_set.keys()))

    with torch.no_grad():
        users_list = []
        ratings_list = []
        groundTruth_items_list = []

        for batch_users in minibatch(test_users, batch_size=args.test_batch_size):
            batch_users = batch_users.to(args.device)
            rating_batch = torch.matmul(user_embs[batch_users], item_embs.t())

            clicked_items = []
            for user in batch_users:
                clicked_items_for_user_in_train = train_user_set[user.item()] - args.n_users
                clicked_items_for_user_in_val = val_user_set[user.item()] - args.n_users
                clicked_items.append(np.concatenate([clicked_items_for_user_in_train, clicked_items_for_user_in_val]))
                
            groundTruth_items = [test_user_set[user.item()] - args.n_users
                                 for user in batch_users]

            exclude_index = []
            exclude_items = []

            for range_i, items in enumerate(clicked_items):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)

            rating_batch[exclude_index, exclude_items] = -(1 << 10)

            rating_K = torch.topk(rating_batch, k=max(args.topks))[1].cpu()

            users_list.append(batch_users.cpu().tolist())
            ratings_list.append(rating_K)
            groundTruth_items_list.append(groundTruth_items)

        X = zip(ratings_list, groundTruth_items_list)

        user_performance_dict = defaultdict(list)
        k = args.topks[0] # only consider one evaluation at a time
        for i, x in enumerate(X):
            sorted_items = x[0].numpy()
            groundTrue = x[1]
            r = getLabel(groundTrue, sorted_items)

            users_batch = users_list[i]
            for iter_ in range(len(groundTrue)): # calculate per user-item pair
                ret = RecallPrecision_ATk([groundTrue[iter_]], r[iter_].reshape(1, -1), k)
                ndcgs = NDCGatK_r([groundTrue[iter_]], r[iter_].reshape(1, -1), k)
                hit_ratios = Hit_at_k(r[iter_].reshape(1, -1), k)

                hit_ratio = sum(hit_ratios)
                pre = sum(ret['Precision'])
                recall = sum(ret['Recall'])
                ndcg = sum(ndcgs)
                temp = ret['Precision'] + ret['Recall']
                temp[temp == 0] = float('inf')
                F1s = 2 * ret['Precision'] * ret['Recall'] / temp
                F1 = sum(F1s)
                
                performance = [recall, ndcg, pre, hit_ratio, F1]
                user_performance_dict[users_batch[iter_]].append(performance)
    if not os.path.exists('./results'):
        os.mkdir('./results')
    if args.flag == 0:
        args.per_user_filename = './results/per_user_{}_{}.pkl'.format(args.dataset, args.model)
    elif args.flag == 1:
        args.per_user_filename = './results/per_user_{}_{}_{}_{}.pkl'.format(args.dataset, args.model, args.strategy, str(args.aug_num))
    elif args.flag == 2:
        args.per_user_filename = './results/user_aug_per_user_{}_{}_{}_{}.pkl'.format(args.dataset, args.model, args.strategy, str(args.aug_num))
    
    with open(args.per_user_filename, 'wb') as f:
        pickle.dump(user_performance_dict, f)

def test_per_item(user_embs, item_embs, user_dict, args):
    train_user_set = user_dict['train_user_set']
    val_user_set = user_dict['val_user_set']
    test_user_set = user_dict['test_user_set']
    test_users = torch.tensor(list(test_user_set.keys()))

    with torch.no_grad():
        users_list = []
        ratings_list = []
        groundTruth_items_list = []

        for batch_users in minibatch(test_users, batch_size=args.test_batch_size):
            batch_users = batch_users.to(args.device)
            rating_batch = torch.matmul(user_embs[batch_users], item_embs.t())

            clicked_items = []
            for user in batch_users:
                clicked_items_for_user_in_train = train_user_set[user.item()] - args.n_users
                clicked_items_for_user_in_val = val_user_set[user.item()] - args.n_users
                clicked_items.append(np.concatenate([clicked_items_for_user_in_train, clicked_items_for_user_in_val]))
                
            groundTruth_items = [test_user_set[user.item()] - args.n_users
                                 for user in batch_users]

            exclude_index = []
            exclude_items = []

            for range_i, items in enumerate(clicked_items):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)

            rating_batch[exclude_index, exclude_items] = -(1 << 10)

            rating_K = torch.topk(rating_batch, k=max(args.topks))[1].cpu()

            users_list.append(batch_users.cpu().tolist())
            ratings_list.append(rating_K)
            groundTruth_items_list.append(groundTruth_items)

        X = zip(ratings_list, groundTruth_items_list)

        item_performance_dict = defaultdict(list)
        k = args.topks[0] # only consider one evaluation at a time
        for i, x in enumerate(X):
            sorted_items = x[0].numpy()
            groundTrue = x[1]

            for iter_ in range(len(groundTrue)): # calculate per user-item pair
                items_cur = groundTrue[iter_]
                for item_cur in items_cur:
                    r = getLabel([[item_cur]], [sorted_items[iter_]])
                    ret = RecallPrecision_ATk([[item_cur]], r[0].reshape(1, -1), k)
                    ndcgs = NDCGatK_r([[item_cur]], r[0].reshape(1, -1), k)
                    hit_ratios = Hit_at_k(r[0].reshape(1, -1), k)
    
                    hit_ratio = sum(hit_ratios)
                    pre = sum(ret['Precision'])
                    recall = sum(ret['Recall'])
                    ndcg = sum(ndcgs)
                    temp = ret['Precision'] + ret['Recall']
                    temp[temp == 0] = float('inf')
                    F1s = 2 * ret['Precision'] * ret['Recall'] / temp
                    F1 = sum(F1s)
                    
                    performance = [recall, ndcg, pre, hit_ratio, F1]
                    item_performance_dict[item_cur].append(performance)

    if not os.path.exists('./results'):
        os.mkdir('./results')
    if args.flag == 0:
        args.per_item_filename = './results/per_item_{}_{}.pkl'.format(args.dataset, args.model)
    elif args.flag == 1:
        args.per_item_filename = './results/per_item_{}_{}_{}_{}.pkl'.format(args.dataset, args.model, args.strategy, str(args.aug_num))
    elif args.flag == 2:
        args.per_item_filename = './results/user_aug_per_item_{}_{}_{}_{}.pkl'.format(args.dataset, args.model, args.strategy, str(args.aug_num))
    with open(args.per_item_filename, 'wb') as f:
        pickle.dump(item_performance_dict, f)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_roc_score(edges_pos, edges_neg, score_matrix, apply_sigmoid=False):
    score_matrix = score_matrix.cpu().detach()
    if len(edges_pos) == 0 or len(edges_neg) == 0:
        return (None, None, None)

    # Store positive edge predictions, actual values
    preds_pos = []
    pos = []
    for edge in edges_pos:
        if apply_sigmoid == True:
            preds_pos.append(sigmoid(score_matrix[int(edge[0]), int(edge[1])]))
        else:
            preds_pos.append(score_matrix[int(edge[0]), int(edge[1])])
        pos.append(1) # actual value (1 for positive)

    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    for edge in edges_neg:
        if apply_sigmoid == True:
            preds_neg.append(sigmoid(score_matrix[int(edge[0]), int(edge[1])]))
        else:
            preds_neg.append(score_matrix[int(edge[0]), int(edge[1])])
        neg.append(0) # actual value (0 for negative)

    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

