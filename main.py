import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from scipy.sparse import csr_matrix
from tqdm import tqdm

from dataset import get_data
from model import ADDVAE
from utils import get_device, write_dict_to_csv, save_to_json
from metrics import recall_at_k_batch, ndcg_binary_at_k_batch


def get_string_config(args):
    excluded_attributes = {'path', 'file_name'}
    if args.file_name is not None:
        selected_attr = [attr for attr in args.file_name.split(',')]
        return f'{args.dataset}_' + '_'.join([str(args.__dict__[attr]) for attr in selected_attr])
    return '_'.join([str(value) for attr, value in args.__dict__.items()
                     if attr not in excluded_attributes])


def train(model, train_loader, test_loader, optimizer, device):
    monitor_metric = 'recall@20'
    best_score = -1
    monitor_metric2 = 'ndcg@20'
    best_score2 = -1
    best_epoch = -1
    num_steps = 0

    if not os.path.isdir('./saved_models/'):
        os.mkdir('./saved_models/')
    model_file = './saved_models/ADDVAE_' + get_string_config(args) + '.mdl'

    for epoch in range(1, args.epochs + 1):
        model.train()
        print(f'Epoch {epoch}/{args.epochs}:')
        running_loss = 0.0
        epoch_size = 0

        for data in tqdm(train_loader, desc='Training'):
            _, batch_ratings, batch_texts = data
            local_batch_size = batch_ratings.shape[0]
            epoch_size += local_batch_size
            batch_ratings = batch_ratings.float().to(device)
            batch_texts = batch_texts.float().to(device)

            loss = model.calculate_loss(input_rating=batch_ratings,
                                        input_text=batch_texts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.data * local_batch_size
        # total loss
        epoch_loss = running_loss / epoch_size
        print(f'Epoch: {epoch}. Loss: {epoch_loss}')

        print('Validation results:')
        valid_results = test(model, test_loader, device, topk=[20])

        if valid_results[monitor_metric] > best_score:
            best_score = valid_results[monitor_metric]
            best_epoch = epoch
            num_steps = 0
            model.save(out_file=model_file)
        elif valid_results[monitor_metric] == best_score and valid_results[monitor_metric2] > best_score2:
            best_score2 = valid_results[monitor_metric2]
            best_epoch = epoch + 1
            num_steps = 0
            model.save(out_file=model_file)
        else:
            num_steps += 1
            if num_steps >= args.early_stopping:
                break
        print(f'Best score: {best_score} / best epoch: {best_epoch}')
        print('*' * 30)
    print(30 * '+' + 'TRAINING COMPLETED!' + 30 * '+')

    return model_file


def test(model, test_loader, device, topk=None):
    with torch.no_grad():
        model.eval()

        result_dict = {}
        if topk is None:
            list_k = [10]
        else:
            list_k = topk

        for data in tqdm(test_loader, desc='Predicting'):
            _, input_ratings, gt_ratings, batch_texts = data
            input_ratings = input_ratings.float().to(device)
            gt_ratings = gt_ratings.to(device)
            batch_texts = batch_texts.float().to(device)

            sum_ratings = gt_ratings.sum(dim=-1)
            selected_indices = sum_ratings.nonzero(as_tuple=True)[0]
            input_ratings = input_ratings[selected_indices]
            gt_ratings = gt_ratings[selected_indices]
            batch_texts = batch_texts[selected_indices]

            batch_prediction, _ = model.predict(input_rating=input_ratings,
                                                input_text=batch_texts)

            batch_prediction = batch_prediction.cpu().numpy()
            gt_ratings = csr_matrix(gt_ratings.cpu().numpy())

            for metric in {'recall', 'ndcg'}:
                for k in list_k:
                    mtk = f'{metric}@{k}'
                    if mtk not in result_dict:
                        result_dict[mtk] = []
                    metric_function = recall_at_k_batch if metric == 'recall' else ndcg_binary_at_k_batch
                    result_dict[mtk] += metric_function(x_pred=batch_prediction,
                                                        heldout_batch=gt_ratings,
                                                        k=k).tolist()
        avg_result_dict = {}
        for key in sorted(result_dict.keys()):
            avg_result_dict[key] = round(np.mean(result_dict[key]), 4)
        for key in avg_result_dict:
            print(f'\t{key}: {avg_result_dict[key]}')

    return avg_result_dict


def predict_cluster(model, train_loader, word_vocab, dataset, device):
    with torch.no_grad():
        model.eval()

        cluster_dists = None
        cluster_scores = None
        user_indices = None
        for data in tqdm(train_loader, desc='Predicting'):
            indices, input_ratings, input_texts = data
            input_ratings = input_ratings.float().to(device)
            input_texts = input_texts.float().to(device)
            _, text_outputs = model.predict_per_prototype_output(input_ratings, input_texts)
            text_outputs = torch.stack(text_outputs, dim=0)
            text_outputs = text_outputs.permute(1, 0, 2)
            top_scores, top_indices = text_outputs.topk(k=20, dim=-1)
            cluster_dists = top_indices if cluster_dists is None else \
                torch.vstack((cluster_dists, top_indices))
            cluster_scores = top_scores if cluster_scores is None else \
                torch.vstack((cluster_scores, top_scores))
            user_indices = indices.unsqueeze(dim=-1) if user_indices is None else \
                torch.vstack((user_indices, indices.unsqueeze(dim=-1)))

        cluster_dist_dict = {}
        idx_to_word = {val: key for key, val in word_vocab.items()}
        cluster_dists = cluster_dists.cpu().numpy().tolist()
        cluster_scores = cluster_scores.cpu().numpy().tolist()
        user_indices = user_indices.squeeze(dim=-1).cpu().numpy().tolist()
        for i, dist in tqdm(zip(user_indices, cluster_dists), desc='Predicting user words'):
            cluster_dist_dict[i] = {}
            for idx in range(len(cluster_dists[i])):
                cluster_dist_dict[i][chr(idx + 65)] = []
                for iidx, widx in enumerate(cluster_dists[i][idx]):
                    cluster_dist_dict[i][chr(idx + 65)].append({idx_to_word[widx]: cluster_scores[i][idx][iidx]})

    save_to_json(cluster_dist_dict,
                 out_file=f'./data/{dataset}/{dataset}.DualMacridVAE_text.{get_string_config(args)}.cluster.dist')


def run(args):
    device = get_device()

    num_workers = 4 if torch.cuda.is_available() else 1
    data_loader, data_stats = get_data(args, binary=True, num_workers=num_workers, train_shuffle=True)
    num_items = data_stats['num_items']
    num_words = data_stats['num_words']

    train_loader, valid_loader, test_loader = data_loader
    num_batches = int(np.ceil(len(train_loader) / args.batch_size))

    total_anneal_steps = num_batches * 5
    if args.rating_total_anneal_steps > 0:
        rating_total_anneal_steps = args.rating_total_anneal_steps
    else:
        rating_total_anneal_steps = total_anneal_steps

    if args.text_total_anneal_steps > 0:
        text_total_anneal_steps = args.text_total_anneal_steps
    else:
        text_total_anneal_steps = total_anneal_steps

    keys = ['num_users', 'num_items', 'num_ratings', 'num_valid_users', 'num_valid_items', 'num_valid_ratings']
    print(f'Dataset: {args.dataset}')
    for key in keys:
        print(f'\t{key}: {data_stats[key]}')

    layers = [int(d) for d in args.layers.split(',')]
    reg_weights = [float(w) for w in args.reg_weights.split(',')]
    model = ADDVAE(num_items=num_items,
                   num_words=num_words,
                   layers=layers,
                   emb_size=args.emb_dim,
                   dropout=args.dropout,
                   num_prototypes=args.num_clusters,
                   tau=args.temperature,
                   nogb=False,
                   rating_anneal_cap=args.rating_anneal_cap,
                   text_anneal_cap=args.text_anneal_cap,
                   std=args.std,
                   text_std=args.text_std,
                   reg_weights=reg_weights,
                   lambda_text=args.lambda_text,
                   lambda_reg=args.lambda_reg,
                   cmode=args.combi_mode,
                   rmode=args.reg_mode,
                   coda=bool(int(args.coda)),
                   scale=bool(int(args.scale)),
                   subtract_mean=bool(int(args.subtract_mean)),
                   rating_total_anneal_steps=rating_total_anneal_steps,
                   text_total_anneal_steps=text_total_anneal_steps,
                   subtract_mean_n=args.subtract_mean_n)
    model = model.to(device)

    params = model.get_params()
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(params=params, lr=args.lr)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(params=params, lr=args.lr)
    else:
        raise Exception('Invalid optimizer')

    topk = [int(k) for k in args.topk.split(',')]

    print('TRAINING STARTED!')
    model_file = train(model, train_loader, valid_loader, optimizer, device)

    print('LOAD SAVED MODEL...')
    trained_model = model.load(model_file).to(device)

    all_results = {}
    print('After training, validation results:')
    valid_results = test(trained_model, valid_loader, device, topk=topk)
    for key in valid_results:
        all_results[f'v_{key}'] = valid_results[key]
    if bool(args.eval_on_test):
        print('Test results:')
        test_results = test(trained_model, test_loader, device, topk=topk)
        for key in test_results:
            all_results[f't_{key}'] = test_results[key]
    if not os.path.isdir('./results'):
        os.mkdir('./results')
    write_dict_to_csv(data_dict=all_results,
                      out_file=f'./results/ADDVAE_{get_string_config(args)}.csv')

    os.remove(model_file)

    data_loader, data_stats = get_data(args, binary=True, num_workers=num_workers, train_shuffle=False)
    train_loader, _, _ = data_loader
    word_vocab = data_stats['word_vocab']
    predict_cluster(trained_model, train_loader, word_vocab, args.dataset, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', '--dataset', type=str, default='citeulike',
                        help='Dataset')
    parser.add_argument('-l', '--layers', type=str, default="200",
                        help='Layer dimensions, multiple dimensions are separated by comma')
    parser.add_argument('-dim', '--emb_dim', type=int, default=64,
                        help='Embedding size')
    parser.add_argument('-ep', '--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('-bs', '--batch_size', type=int, default=512,
                        help='Batch data size in training/testing')
    parser.add_argument('-dr', '--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('-optim', '--optimizer', type=str, default='Adam',
                        help='Optimization algorithm')
    parser.add_argument('-lr', '--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('-ncl', '--num_clusters', type=int, default=4,
                        help='Number of clusters, i.e., number of prototypes or number of user interests')
    parser.add_argument('-tau', '--temperature', type=float, default=0.1,
                        help='Temperature value')
    parser.add_argument('-ranneal', '--rating_anneal_cap', type=float, default=0.2,
                        help='Annealing value for KL divergence for rating')
    parser.add_argument('-tanneal', '--text_anneal_cap', type=float, default=0.2,
                        help='Annealing value for KL divergence for text')
    parser.add_argument('-std', '--std', type=float, default=0.075,
                        help='Standard deviation of Gaussian prior for rating')
    parser.add_argument('-tstd', '--text_std', type=float, default=0.075,
                        help='Standard deviation of Gaussian prior for text')
    parser.add_argument('-reg', '--reg_weights', type=str, default="0,0",
                        help='Regularization weights')
    parser.add_argument('-fname', '--file_name', type=str, default=None,
                        help='Selected params from args to construct file name, separated by comma')
    parser.add_argument('-gpu', '--gpu_id', type=int, default=2,
                        help='GPU ID')
    parser.add_argument('-mdl', '--model_name', type=str, default='ADDVAE',
                        help='Model name')
    parser.add_argument('-topk', '--topk', type=str, default='10,20,50',
                        help='List of top-k to evaluate, separated by comma')
    parser.add_argument('-eot', '--eval_on_test', type=int, default=1,
                        help='1 to evaluate on test set. 0 to only evaluate on validation set')
    parser.add_argument('-rs', '--random_seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('-ldr_text', '--lambda_text', type=float, default=1,
                        help='Lambda in loss term for text')
    parser.add_argument('-ldr_reg', '--lambda_reg', type=float, default=0.1,
                        help='Lambda in loss term for text for regularization')
    parser.add_argument('-cmode', '--combi_mode', type=str, default='attention',
                        help='Method to combine factors from ratings and texts, '
                             'i.e., attention (ADDVAE_MI) or regularization (ADDVAE_dist)')
    parser.add_argument('-rmode', '--reg_mode', type=str, default='distance',
                        help='Regularization method, i.e., mutual_info (ADDVAE_MI), distance (ADDVAE_dist)')
    parser.add_argument('-coda', '--coda', type=int, default=1,
                        help='1 to turn on CODA network. 0 to use vanilla attention')
    parser.add_argument('-scale', '--scale', type=int, default=1,
                        help='1 to use scaling factor in attention')
    parser.add_argument('-rtast', '--rating_total_anneal_steps', type=int, default=20000,
                        help='Total Anneal Steps for rating')
    parser.add_argument('-ttast', '--text_total_anneal_steps', type=int, default=20000,
                        help='Total Anneal Steps for text')
    parser.add_argument('-es', '--early_stopping', type=int, default=5,
                        help='Number of steps for early stopping')
    parser.add_argument('-stm', '--subtract_mean', type=int, default=1,
                        help='1 to use subtract mean matrix E in CoDA. 0 to use multiplication by 2')
    parser.add_argument('-stmn', '--subtract_mean_n', default=0, type=int,
                        help='1 to use subtract mean matrix N in CoDA. 0 to use multiplication by 2')
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    os.environ['CUDA_DEVICE'] = str(args.gpu_id)
    run(args)
