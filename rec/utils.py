"""Utility functions for training and evaluation."""

import os
import random

import numpy as np
import torch
from torch import nn
from torch_scatter import scatter_add, scatter_max
import matplotlib.pyplot as plt


def neg_sample_before_epoch(train_cf, clicked_set, args):
    """Sample negative items for each training edge.

    Args:
        train_cf (np.ndarray): Training edges.
        clicked_set (dict): User-to-clicked items mapping.
        args (argparse.Namespace): Parsed arguments.

    Returns:
        np.ndarray: Negative samples per training edge.
    """
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
    """Move a batch dict of tensors to a device.

    Args:
        batch (dict): Batch tensor dictionary.
        device (torch.device): Target device.

    Returns:
        dict: Batch on target device.
    """
    for c in batch:
        batch[c] = batch[c].to(device)

    return batch


def seed_everything(seed):
    """Seed Python, NumPy, and PyTorch RNGs.

    Args:
        seed (int): Random seed.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def minibatch(*tensors, batch_size):
    """Yield minibatches from one or more tensors.

    Args:
        *tensors: One or more tensors or arrays with matching length.
        batch_size (int): Batch size.

    Yields:
        tuple or array: Minibatch slice(s).
    """
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def knn_adj(adj_sp_norm, args):
    """Build a k-NN filtered adjacency from a dense adjacency.

    Args:
        adj_sp_norm (torch.Tensor): Sparse adjacency.
        args (argparse.Namespace): Parsed arguments with knn.

    Returns:
        torch.sparse.FloatTensor: Filtered adjacency.
    """
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
    """Compute per-edge inverse user degree ratios.

    Args:
        train_cf (np.ndarray): Training edges.
        n_users (int): Number of users.

    Returns:
        torch.Tensor: Ratios indexed by edges.
    """
    user_link_num = torch.tensor(
        [(train_cf[:, 0] == i).sum() for i in range(n_users)])

    link_ratio = n_users / user_link_num
    link_ratio = link_ratio / link_ratio.sum()

    return link_ratio[train_cf[:, 0]]


def cal_bpr_loss(user_embs, pos_item_embs, neg_item_embs, link_ratios=None):
    """Compute BPR loss.

    Args:
        user_embs (torch.Tensor): User embeddings.
        pos_item_embs (torch.Tensor): Positive item embeddings.
        neg_item_embs (torch.Tensor): Negative item embeddings.
        link_ratios (torch.Tensor, optional): Unused.

    Returns:
        torch.Tensor: Scalar loss.
    """
    pos_scores = torch.sum(torch.mul(user_embs, pos_item_embs), axis=1)
    neg_scores = torch.sum(torch.mul(user_embs.unsqueeze(dim=1), neg_item_embs), axis=-1)
    # modify the loss to the original bpr loss in lightgcn
    bpr_loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
    return bpr_loss


def cal_l2_loss(user_embs, pos_item_embs, neg_item_embs, batch_size):
    """Compute L2 regularization loss.

    Args:
        user_embs (torch.Tensor): User embeddings.
        pos_item_embs (torch.Tensor): Positive item embeddings.
        neg_item_embs (torch.Tensor): Negative item embeddings.
        batch_size (int): Batch size.

    Returns:
        torch.Tensor: Scalar L2 loss.
    """
    return 0.5 * (user_embs.norm(2).pow(2) + pos_item_embs.norm(2).pow(2) + neg_item_embs.norm(2).pow(2)) / batch_size


def softmax(src, index, num_nodes):
    """Compute sparse softmax over segments.

    Args:
        src (torch.Tensor): Source values.
        index (torch.Tensor): Indices for segments.
        num_nodes (int): Number of nodes.

    Returns:
        torch.Tensor: Softmax-normalized values.
    """
    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (
        scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out
