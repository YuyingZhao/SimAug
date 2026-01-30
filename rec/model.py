"""Model definitions for recommendation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from utils import softmax


class GraphConv(nn.Module):
    """Graph convolutional layer stack for LightGCN."""

    def __init__(self, args):
        """Initialize the graph convolution module.

        Args:
            args (argparse.Namespace): Parsed arguments.
        """
        super(GraphConv, self).__init__()
        self.args = args

    def forward(self, embed, adj_sp_norm, edge_index, edge_weight, deg):
        """Run message passing for multiple hops.

        Args:
            embed (torch.Tensor): Node embeddings.
            adj_sp_norm (torch.Tensor): Normalized adjacency (unused).
            edge_index (torch.Tensor): Edge indices.
            edge_weight (torch.Tensor): Edge weights.
            deg (torch.Tensor): Node degrees (unused).

        Returns:
            tuple: (user_embs, item_embs) stacked by hop.
        """
        agg_embed = embed
        embs = [embed]

        row, col = edge_index

        for hop in range(self.args.n_hops):
            out = agg_embed[row] * edge_weight.unsqueeze(-1)
            agg_embed = scatter(
                out, col, dim=0, dim_size=self.args.n_users + self.args.n_items, reduce='add')
            embs.append(agg_embed)

        embs = torch.stack(embs, dim=1)  # [n_entity, n_hops+1, emb_size]

        return embs[:self.args.n_users, :], embs[self.args.n_users:, :]


class LightGCN(nn.Module):
    def __init__(self, args):
        """Initialize LightGCN model.

        Args:
            args (argparse.Namespace): Parsed arguments.
        """
        super(LightGCN, self).__init__()

        self.args = args

        self._init_weight()
        self.gcn = self._init_model()

    def _init_weight(self):
        """Initialize ID embeddings."""
        initializer = nn.init.xavier_uniform_
        self.embeds = nn.Parameter(initializer(torch.empty(
            self.args.n_users + self.args.n_items, self.args.embedding_dim)))

    def _init_model(self):
        """Construct the graph convolution module."""
        if self.args.model == 'LightGCN':
            return GraphConv(self.args)

    def batch_generate(self, user, pos_item, neg_item):
        """Generate embeddings for a training batch.

        Args:
            user (torch.Tensor): User indices.
            pos_item (torch.Tensor): Positive item indices.
            neg_item (torch.Tensor): Negative item indices.

        Returns:
            tuple: (user_embs, pos_item_embs, neg_item_embs)
        """
        user_gcn_embs, item_gcn_embs = self.gcn(
            self.embeds, self.adj_sp_norm, self.edge_index, self.edge_weight, self.deg)

        user_gcn_embs, item_gcn_embs = self.pooling(
            user_gcn_embs), self.pooling(item_gcn_embs)

        user_embs = user_gcn_embs[user]
        pos_item_embs = item_gcn_embs[pos_item - self.args.n_users]
        neg_item_embs = item_gcn_embs[neg_item - self.args.n_users]

        return user_embs, pos_item_embs, neg_item_embs

    def forward(self, batch=None):
        """Forward pass for training.

        Args:
            batch (dict): Batch dictionary with users/items.

        Returns:
            tuple: Embeddings for loss computation.
        """
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        user_embs, pos_item_embs, neg_item_embs = self.batch_generate(
            user, pos_item, neg_item)

        return user_embs, pos_item_embs, neg_item_embs, self.embeds[user], self.embeds[pos_item], self.embeds[neg_item]

    def pooling(self, embeddings):
        """Aggregate multi-hop embeddings.

        Args:
            embeddings (torch.Tensor): Stacked embeddings.

        Returns:
            torch.Tensor: Pooled embeddings.
        """
        if self.args.aggr == 'mean':
            return embeddings.mean(dim=1)
        elif self.args.aggr == 'sum':
            return embeddings.sum(dim=1)
        elif self.args.aggr == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def generate(self):
        """Generate pooled user and item embeddings.

        Returns:
            tuple: (user_embs, item_embs)
        """
        user_gcn_embs, item_gcn_embs = self.gcn(
            self.embeds, self.adj_sp_norm, self.edge_index, self.edge_weight, self.deg)

        user_embs, item_embs = self.pooling(
            user_gcn_embs), self.pooling(item_gcn_embs)

        return user_embs, item_embs

    def generate_layers(self):
        """Return per-hop embeddings."""
        return self.gcn(self.embeds, self.adj_sp_norm, self.edge_index, self.edge_weight, self.deg)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layer, dropout_prob):
        """Initialize a simple MLP.

        Args:
            input_size (int): Input dimension.
            hidden_size (int): Hidden dimension.
            output_size (int): Output dimension.
            num_layer (int): Number of layers.
            dropout_prob (float): Dropout probability.
        """
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_layer-2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        """Forward pass through the MLP.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Output features.
        """
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x
        
class LightGCN_with_content(nn.Module):
    def __init__(self, args):
        """Initialize LightGCN with content embeddings.

        Args:
            args (argparse.Namespace): Parsed arguments.
        """
        super(LightGCN_with_content, self).__init__()

        self.args = args

        self.gcn = self._init_model()

        # load content embeddings for users and items
        prefix = '../datasets/{}/preprocessed/'.format(args.dataset)
        user_emb_filename = prefix + 'user_embs_sentence_transformer_6.pth'
        user_embs = torch.load(user_emb_filename)
        user_embs = torch.from_numpy(user_embs)
        user_embs = user_embs.to(args.device)

        item_emb_filename = prefix + 'embs_sentence_transformer_6.pth'
        item_embs = torch.load(item_emb_filename)
        item_embs = torch.from_numpy(item_embs)
        item_embs = item_embs.to(args.device)

        self.user_item_embeds = torch.cat([user_embs, item_embs])
        print('user item embedding shape:', self.user_item_embeds.shape)

        self.mlp = MLP(user_embs.shape[1], args.embedding_dim*2, args.embedding_dim, args.mlp_num_layer, args.mlp_dropout_prob)

    def _init_model(self):
        """Construct the graph convolution module."""
        if self.args.model == 'LightGCN_with_content':
            return GraphConv(self.args)

    def batch_generate(self, user, pos_item, neg_item):
        """Generate embeddings for a training batch.

        Args:
            user (torch.Tensor): User indices.
            pos_item (torch.Tensor): Positive item indices.
            neg_item (torch.Tensor): Negative item indices.

        Returns:
            tuple: (user_embs, pos_item_embs, neg_item_embs)
        """
        self.embeds = self.mlp(self.user_item_embeds)
        user_gcn_embs, item_gcn_embs = self.gcn(
            self.embeds, self.adj_sp_norm, self.edge_index, self.edge_weight, self.deg)

        user_gcn_embs, item_gcn_embs = self.pooling(
            user_gcn_embs), self.pooling(item_gcn_embs)

        user_embs = user_gcn_embs[user]
        pos_item_embs = item_gcn_embs[pos_item - self.args.n_users]
        neg_item_embs = item_gcn_embs[neg_item - self.args.n_users]

        return user_embs, pos_item_embs, neg_item_embs

    def forward(self, batch=None):
        """Forward pass for training.

        Args:
            batch (dict): Batch dictionary with users/items.

        Returns:
            tuple: Embeddings for loss computation.
        """
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        user_embs, pos_item_embs, neg_item_embs = self.batch_generate(
            user, pos_item, neg_item)

        return user_embs, pos_item_embs, neg_item_embs, self.embeds[user], self.embeds[pos_item], self.embeds[neg_item]

    def pooling(self, embeddings):
        """Aggregate multi-hop embeddings.

        Args:
            embeddings (torch.Tensor): Stacked embeddings.

        Returns:
            torch.Tensor: Pooled embeddings.
        """
        if self.args.aggr == 'mean':
            return embeddings.mean(dim=1)
        elif self.args.aggr == 'sum':
            return embeddings.sum(dim=1)
        elif self.args.aggr == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def generate(self):
        """Generate pooled user and item embeddings.

        Returns:
            tuple: (user_embs, item_embs)
        """
        self.embeds = self.mlp(self.user_item_embeds)
        user_gcn_embs, item_gcn_embs = self.gcn(
            self.embeds, self.adj_sp_norm, self.edge_index, self.edge_weight, self.deg)

        user_embs, item_embs = self.pooling(
            user_gcn_embs), self.pooling(item_gcn_embs)

        return user_embs, item_embs


class LightGCN_with_content_id_add(nn.Module):
    def __init__(self, args):
        """Initialize LightGCN with content + ID embedding addition.

        Args:
            args (argparse.Namespace): Parsed arguments.
        """
        super(LightGCN_with_content_id_add, self).__init__()

        self.args = args

        self.gcn = self._init_model()

        # load content embeddings for users and items
        prefix = '../datasets/{}/preprocessed/'.format(args.dataset)
        user_emb_filename = prefix + 'user_embs_sentence_transformer_6.pth'
        user_embs = torch.load(user_emb_filename)
        user_embs = torch.from_numpy(user_embs)
        user_embs = user_embs.to(args.device)

        item_emb_filename = prefix + 'embs_sentence_transformer_6.pth'
        item_embs = torch.load(item_emb_filename)
        item_embs = torch.from_numpy(item_embs)
        item_embs = item_embs.to(args.device)

        self.user_item_embeds = torch.cat([user_embs, item_embs])
        print('user item embedding shape:', self.user_item_embeds.shape)

        initializer = nn.init.xavier_uniform_
        self.id_embeds = nn.Parameter(initializer(torch.empty(self.args.n_users + self.args.n_items, self.args.embedding_dim)))

        self.mlp1 = MLP(user_embs.shape[1], args.embedding_dim*2, args.embedding_dim, args.mlp_num_layer, args.mlp_dropout_prob)
        self.mlp2 = MLP(args.embedding_dim, args.embedding_dim, args.embedding_dim, 2, args.mlp_dropout_prob)
        

    def _init_model(self):
        """Construct the graph convolution module."""
        if self.args.model == 'LightGCN_with_content_id_add':
            return GraphConv(self.args)

    def batch_generate(self, user, pos_item, neg_item):
        """Generate embeddings for a training batch.

        Args:
            user (torch.Tensor): User indices.
            pos_item (torch.Tensor): Positive item indices.
            neg_item (torch.Tensor): Negative item indices.

        Returns:
            tuple: (user_embs, pos_item_embs, neg_item_embs)
        """
        self.content_embeds = self.mlp1(self.user_item_embeds)
        self.embeds = self.content_embeds + self.id_embeds
        self.embeds = self.mlp2(self.embeds)
        user_gcn_embs, item_gcn_embs = self.gcn(
            self.embeds, self.adj_sp_norm, self.edge_index, self.edge_weight, self.deg)

        user_gcn_embs, item_gcn_embs = self.pooling(
            user_gcn_embs), self.pooling(item_gcn_embs)

        user_embs = user_gcn_embs[user]
        pos_item_embs = item_gcn_embs[pos_item - self.args.n_users]
        neg_item_embs = item_gcn_embs[neg_item - self.args.n_users]

        return user_embs, pos_item_embs, neg_item_embs

    def forward(self, batch=None):
        """Forward pass for training.

        Args:
            batch (dict): Batch dictionary with users/items.

        Returns:
            tuple: Embeddings for loss computation.
        """
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        user_embs, pos_item_embs, neg_item_embs = self.batch_generate(
            user, pos_item, neg_item)

        return user_embs, pos_item_embs, neg_item_embs, self.embeds[user], self.embeds[pos_item], self.embeds[neg_item]

    def pooling(self, embeddings):
        """Aggregate multi-hop embeddings.

        Args:
            embeddings (torch.Tensor): Stacked embeddings.

        Returns:
            torch.Tensor: Pooled embeddings.
        """
        if self.args.aggr == 'mean':
            return embeddings.mean(dim=1)
        elif self.args.aggr == 'sum':
            return embeddings.sum(dim=1)
        elif self.args.aggr == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def generate(self):
        """Generate pooled user and item embeddings.

        Returns:
            tuple: (user_embs, item_embs)
        """
        self.content_embeds = self.mlp1(self.user_item_embeds)
        self.embeds = self.content_embeds + self.id_embeds
        self.embeds = self.mlp2(self.embeds)
        user_gcn_embs, item_gcn_embs = self.gcn(
            self.embeds, self.adj_sp_norm, self.edge_index, self.edge_weight, self.deg)

        user_embs, item_embs = self.pooling(
            user_gcn_embs), self.pooling(item_gcn_embs)

        return user_embs, item_embs


class LightGCN_with_content_id_cat(nn.Module):
    def __init__(self, args):
        """Initialize LightGCN with content + ID embedding concatenation.

        Args:
            args (argparse.Namespace): Parsed arguments.
        """
        super(LightGCN_with_content_id_cat, self).__init__()

        self.args = args

        self.gcn = self._init_model()

        # load content embeddings for users and items
        prefix = '../datasets/{}/preprocessed/'.format(args.dataset)
        user_emb_filename = prefix + 'user_embs_sentence_transformer_6.pth'
        user_embs = torch.load(user_emb_filename)
        user_embs = torch.from_numpy(user_embs)
        user_embs = user_embs.to(args.device)

        item_emb_filename = prefix + 'embs_sentence_transformer_6.pth'
        item_embs = torch.load(item_emb_filename)
        item_embs = torch.from_numpy(item_embs)
        item_embs = item_embs.to(args.device)

        self.user_item_embeds = torch.cat([user_embs, item_embs])
        print('user item embedding shape:', self.user_item_embeds.shape)

        initializer = nn.init.xavier_uniform_
        self.id_embeds = nn.Parameter(initializer(torch.empty(self.args.n_users + self.args.n_items, self.args.embedding_dim)))

        self.mlp1 = MLP(user_embs.shape[1], args.embedding_dim*2, args.embedding_dim, args.mlp_num_layer, args.mlp_dropout_prob)
        self.mlp2 = MLP(args.embedding_dim*2, args.embedding_dim, args.embedding_dim, 2, args.mlp_dropout_prob)
        

    def _init_model(self):
        """Construct the graph convolution module."""
        if self.args.model == 'LightGCN_with_content_id_cat':
            return GraphConv(self.args)

    def batch_generate(self, user, pos_item, neg_item):
        """Generate embeddings for a training batch.

        Args:
            user (torch.Tensor): User indices.
            pos_item (torch.Tensor): Positive item indices.
            neg_item (torch.Tensor): Negative item indices.

        Returns:
            tuple: (user_embs, pos_item_embs, neg_item_embs)
        """
        self.content_embeds = self.mlp1(self.user_item_embeds)
        self.embeds = torch.cat([self.content_embeds, self.id_embeds], 1)
        self.embeds = self.mlp2(self.embeds)
        user_gcn_embs, item_gcn_embs = self.gcn(
            self.embeds, self.adj_sp_norm, self.edge_index, self.edge_weight, self.deg)

        user_gcn_embs, item_gcn_embs = self.pooling(
            user_gcn_embs), self.pooling(item_gcn_embs)

        user_embs = user_gcn_embs[user]
        pos_item_embs = item_gcn_embs[pos_item - self.args.n_users]
        neg_item_embs = item_gcn_embs[neg_item - self.args.n_users]

        return user_embs, pos_item_embs, neg_item_embs

    def forward(self, batch=None):
        """Forward pass for training.

        Args:
            batch (dict): Batch dictionary with users/items.

        Returns:
            tuple: Embeddings for loss computation.
        """
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        user_embs, pos_item_embs, neg_item_embs = self.batch_generate(
            user, pos_item, neg_item)

        return user_embs, pos_item_embs, neg_item_embs, self.embeds[user], self.embeds[pos_item], self.embeds[neg_item]

    def pooling(self, embeddings):
        """Aggregate multi-hop embeddings.

        Args:
            embeddings (torch.Tensor): Stacked embeddings.

        Returns:
            torch.Tensor: Pooled embeddings.
        """
        if self.args.aggr == 'mean':
            return embeddings.mean(dim=1)
        elif self.args.aggr == 'sum':
            return embeddings.sum(dim=1)
        elif self.args.aggr == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def generate(self):
        """Generate pooled user and item embeddings.

        Returns:
            tuple: (user_embs, item_embs)
        """
        self.content_embeds = self.mlp1(self.user_item_embeds)
        self.embeds = torch.cat([self.content_embeds, self.id_embeds], 1)
        self.embeds = self.mlp2(self.embeds)
        user_gcn_embs, item_gcn_embs = self.gcn(
            self.embeds, self.adj_sp_norm, self.edge_index, self.edge_weight, self.deg)

        user_embs, item_embs = self.pooling(
            user_gcn_embs), self.pooling(item_gcn_embs)

        return user_embs, item_embs

class MF(nn.Module):
    def __init__(self, args):
        """Initialize matrix factorization model.

        Args:
            args (argparse.Namespace): Parsed arguments.
        """
        super(MF, self).__init__()
        self.args = args
        self._init_weight()

    def _init_weight(self):
        """Initialize ID embeddings."""
        initializer = nn.init.xavier_uniform_
        self.embeds = nn.Parameter(initializer(torch.empty(self.args.n_users + self.args.n_items, self.args.embedding_dim)))

    def generate(self):
        """Return user and item embeddings.

        Returns:
            tuple: (user_embs, item_embs)
        """
        user_embs, item_embs = self.embeds[:self.args.n_users,:], self.embeds[self.args.n_users:, :]
        return user_embs, item_embs

    def batch_generate(self, user, pos_item, neg_item):
        """Generate embeddings for a training batch.

        Args:
            user (torch.Tensor): User indices.
            pos_item (torch.Tensor): Positive item indices.
            neg_item (torch.Tensor): Negative item indices.

        Returns:
            tuple: (user_embs, pos_item_embs, neg_item_embs)
        """
        user_embs = self.embeds[user]
        pos_item_embs = self.embeds[pos_item]
        neg_item_embs = self.embeds[neg_item]

        return user_embs, pos_item_embs, neg_item_embs

    def forward(self, batch):
        """Forward pass for training.

        Args:
            batch (dict): Batch dictionary with users/items.

        Returns:
            tuple: Embeddings for loss computation.
        """
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']  # [batch_size, n_negs * K]

        batch_size = user.shape[0]

        user_embs, pos_item_embs, neg_item_embs = self.batch_generate(user, pos_item, neg_item)

        return user_embs, pos_item_embs, neg_item_embs, user_embs, pos_item_embs, neg_item_embs
