import numpy as np
import random
import torch
from torch import nn
from torch_scatter import scatter_max, scatter_add
import matplotlib.pyplot as plt
import os


def neg_sample_before_epoch(train_cf, clicked_set, args):
    neg_cf = np.random.randint(
        args.n_users, args.n_users + args.n_items, (train_cf.shape[0], args.K))

    for i in range(train_cf.shape[0]):
        # neg items will not include in user_clicked_set
        user_clicked_set = clicked_set[train_cf[i, 0]]

        for j in range(args.K):
            while(neg_cf[i, j] in user_clicked_set):
                neg_cf[i, j] = np.random.randint(
                    args.n_users, args.n_users + args.n_items)

    return neg_cf


def batch_to_gpu(batch, device):
    for c in batch:
        batch[c] = batch[c].to(device)

    return batch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def minibatch(*tensors, batch_size):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def knn_adj(adj_sp_norm, args):
    adj_sp_norm = adj_sp_norm.to_dense()
    top_adj_sp_norm, _ = torch.topk(adj_sp_norm, args.knn)
    low, high = top_adj_sp_norm[:, -1], top_adj_sp_norm[:, 0]
    mask = ((adj_sp_norm >= low.unsqueeze(1)) *
            (adj_sp_norm <= high.unsqueeze(1))).float()
    adj_sp_norm = adj_sp_norm * mask
    adj_sp_norm = torch.triu(adj_sp_norm, diagonal=1)

    edge_index = adj_sp_norm.nonzero()
    adj_sp_norm = torch.sparse.FloatTensor(
        edge_index.t(), adj_sp_norm[edge_index[:, 0], edge_index[:, 1]], (args.n_users + args.n_items, args.n_users + args.n_items))

    return adj_sp_norm


def ratio(train_cf, n_users):
    user_link_num = torch.tensor(
        [(train_cf[:, 0] == i).sum() for i in range(n_users)])

    link_ratio = n_users / user_link_num
    link_ratio = link_ratio / link_ratio.sum()

    return link_ratio[train_cf[:, 0]]


def cal_bpr_loss(user_embs, pos_item_embs, neg_item_embs, link_ratios=None):
    pos_scores = torch.sum(torch.mul(user_embs, pos_item_embs), axis=1)
    neg_scores = torch.sum(torch.mul(user_embs.unsqueeze(dim=1), neg_item_embs), axis=-1)
    # modify the loss to the original bpr loss in lightgcn
    bpr_loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
    return bpr_loss


def cal_l2_loss(user_embs, pos_item_embs, neg_item_embs, batch_size):
    return 0.5 * (user_embs.norm(2).pow(2) + pos_item_embs.norm(2).pow(2) + neg_item_embs.norm(2).pow(2)) / batch_size


def softmax(src, index, num_nodes):
    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (
        scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out