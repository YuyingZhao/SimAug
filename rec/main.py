"""Training entry point for recommendation models."""

import os
from parse import parse_args
from torch.utils.data import DataLoader
from prettytable import PrettyTable
import time
from utils import *
from evaluation import *
from model import *
from dataprocess import *


def run(model, optimizer, train_cf, clicked_set, user_dict, adj, args):
    """Train and validate a model with early stopping.

    Args:
        model (torch.nn.Module): Recommendation model.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        train_cf (np.ndarray): Training user-item edges.
        clicked_set (dict): User-to-clicked items mapping.
        user_dict (dict): Train/val/test user sets.
        adj (torch.Tensor): Edge index for adjacency.
        args (argparse.Namespace): Parsed arguments.

    Returns:
        None
    """
    test_recall_best, early_stop_count = -float('inf'), 0

    adj_sp_norm, deg = normalize_edge(adj, args.n_users, args.n_items)
    edge_index, edge_weight = adj_sp_norm._indices(), adj_sp_norm._values()

    model.adj_sp_norm = adj_sp_norm.to(args.device)
    model.edge_index = edge_index.to(args.device)
    model.edge_weight = edge_weight.to(args.device)
    model.deg = deg.to(args.device)

    args.user_dict = user_dict

    start = time.time()
    total_losses = []

    for epoch in range(args.epochs):
        neg_cf = neg_sample_before_epoch(train_cf, clicked_set, args)
        dataset = Dataset(users=train_cf[:, 0], pos_items=train_cf[:, 1], neg_items=neg_cf, args=args)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                collate_fn=dataset.collate_batch, pin_memory=args.pin_memory)  

        """training"""
        model.train()
        loss = 0

        batch_losses = []
        for i, batch in enumerate(dataloader):
            batch = batch_to_gpu(batch, args.device)
            user_embs, pos_item_embs, neg_item_embs, user_embs0, pos_item_embs0, neg_item_embs0 = model(batch)
            bpr_loss = cal_bpr_loss(user_embs, pos_item_embs, neg_item_embs)
            l2_loss = cal_l2_loss(user_embs0, pos_item_embs0, neg_item_embs0, user_embs0.shape[0])
            batch_loss = bpr_loss + args.l2 * l2_loss
            batch_losses.append(batch_loss.item())
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()

        #******************evaluation****************
        total_losses.append(loss / (i + 1))

        if not epoch % 5:
            model.eval()
            res = PrettyTable()
            res.field_names = ["Time", "Epoch", "Training_loss",
                               "Recall", "NDCG", "Precision", "Hit_ratio", "F1"]
            user_embs, item_embs = model.generate()
            test_res = test(user_embs, item_embs, user_dict, args, flag='val')
            res.add_row(
                [format(time.time() - start, '.4f'), epoch, format(loss / (i + 1), '.4f'), round(test_res['Recall'][0], 4), round(test_res['NDCG'][0], 4), round(test_res['Precision'][0], 4), round(test_res['Hit_ratio'][0], 4), round(test_res['F1'][0], 4)])
            print(res)

            # *********************************************************
            utility_measurement = (test_res['Recall'][0]+test_res['NDCG'][0]+test_res['Precision'][0]+test_res['Hit_ratio'][0]+test_res['F1'][0])/5

            if utility_measurement > test_recall_best:
                test_recall_best = utility_measurement
                early_stop_count = 0
                if args.save:
                    torch.save(model.state_dict(), args.save_model_filename)
            else:
                early_stop_count += 1

            if early_stop_count >= args.early_stop:
                print('Early stop at:', epoch)
                break

    print("Best validation: ", round(test_recall_best, 4))

if __name__ == '__main__':
    args = parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.dataset_path = '../datasets/{}/'.format(args.dataset)
    if args.flag == 0:
        args.save_model_filename = './trained_models/{}/{}_{}.pkl'.format(args.dataset, args.model, str(args.seed))
    elif args.flag == 1: # add items for inactive users
        args.save_model_filename = './trained_models/{}/aug_{}_{}_{}_{}.pkl'.format(args.dataset, args.model, args.strategy, str(args.aug_num), str(args.seed))
    elif args.flag == 2: # add users for unpopular items
        args.save_model_filename = './trained_models/{}/aug_user_{}_{}_{}_{}.pkl'.format(args.dataset, args.model, args.strategy, str(args.aug_num), str(args.seed))
    seed_everything(args.seed)
    print("Configurations:", args)

    """build dataset"""
    train_cf, val_cf, test_cf, user_dict, args.n_users, args.n_items, clicked_set, adj = load_data(args)

    print(args.n_users, args.n_items, train_cf.shape[0] + val_cf.shape[0], test_cf.shape[0],
        round((train_cf.shape[0] + val_cf.shape[0] + test_cf.shape[0]) / (args.n_items * args.n_users), 4))

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

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # create dictionaries if not exists (trained_models/dataset/)
    
    paths = ["./trained_models/", "./trained_models/"+args.dataset]
    for p in paths: 
        if not os.path.exists(p):
            os.mkdir(p)
            print("path has been created: ", p)

    run(model, optimizer, train_cf, clicked_set, user_dict, adj, args)
