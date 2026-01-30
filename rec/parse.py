"""Argument parsing for training and evaluation runs."""

import argparse

def parse_args():
    """Parse CLI arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--dataset", type=str, default="Appliances",
                        help="The name of the dataset")

    # model
    parser.add_argument("--model", type=str, default="MF",
                        help="Choose a recommender:[MF, LightGCN, NGCF]")
    parser.add_argument("--n_hops", type=int, default=2)
    parser.add_argument("--aggr", type=str, default='mean')
    parser.add_argument("--layer_sizes", nargs='?', default=[64, 64, 64])

    # training
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of pretraining epochs')
    parser.add_argument('--batch_size', type=int,
                        default=2048, help='batch size')
    parser.add_argument('--num_workers', type=int,
                        default=12, help='cpu workers to sample for preparing batch')
    parser.add_argument('--pin_memory', type=int, default=1)
    # parser.add_argument('--test_batch_size', type=int,
    #                     default=1024, help='batch size in evaluation phase')
    parser.add_argument('--test_batch_size', type=int,
                        default=2048, help='batch size in evaluation phase')
    parser.add_argument('--embedding_dim', type=int,
                        default=64, help='embedding size')
    parser.add_argument('--num_clusters', type=int,
                        default=16, help='cluster size')
    parser.add_argument('--l2', type=float, default=1e-4,
                        help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--trend_coeff', type=float,
                        default=2, help='coefficient of attention')
    parser.add_argument('--type', type=str,
                        default='jc', help='coefficient of attention')

    # recommendation
    parser.add_argument("--K", type=int, default=1,
                        help="number of negative in K-pair loss")
    parser.add_argument('--topks', nargs='+', type=int, default=[20])
    parser.add_argument("--neg_in_val_test", type=int, default=0,
                        help="whether negative samples could come from validation and testing data, 0 means allow, 1 means not allow")

    # experiments
    parser.add_argument("--save", type=int, default=1,
                        help="save model or not")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed to run the experiment")
    parser.add_argument("--early_stop", type=int, default=10,
                        help="early_stopping by which epoch*5")


    parser.add_argument('--flag', type=int, default=0)
    parser.add_argument('--aug_num', type=int, default=1) # used for augmenting one side
    # if augmenting both side, use these two
    parser.add_argument('--aug_user_num', type=int, default=1) # aug how many users for item side
    parser.add_argument('--aug_item_num', type=int, default=1)
    parser.add_argument('--strategy', type=str, default='random')
    parser.add_argument('--user_item', type=int, default=0) # 0: aug for inactive users, 1: aug for unpopular items, 2: both
    parser.add_argument('--mlp_num_layer', type=int, default=2)
    parser.add_argument('--mlp_dropout_prob', type=float, default=0.3)
    return parser.parse_args()
