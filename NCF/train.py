import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
import time

from tqdm import tqdm

from model import NCF
from data_loader import load_data, get_records
import copy


def train(args, is_topk):
    data = load_data(args)
    pre_entity_embedding = data[0]
    train_set, eval_set, test_set = data[1], data[2], data[3]
    rec = data[4]
    test_records = get_records(test_set)
    n_entity = pre_entity_embedding.shape[0]
    # t.manual_seed(255)
    x = t.randn(n_entity, args.dim)
    # embedding_matrix = t.cat([x, pre_entity_embedding], dim=1)
    embedding_matrix = pre_entity_embedding
    if t.cuda.is_available():
        embedding_matrix = embedding_matrix.to(args.device)
    model = NCF(args.dim)
    criterion = nn.BCELoss()

    if t.cuda.is_available():
        model = model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    print(args.dataset + '-----------------------------------------')

    print('dim: %d' % args.dim, end='\t')
    print('lr: %1.0e' % args.lr, end='\t')
    print('l2: %1.0e' % args.l2, end='\t')
    print('batch_size: %d' % args.batch_size)

    eval_auc_list = []
    test_auc_list = []
    test_acc_list = []
    models = []
    for epoch in (range(args.epochs)):
        start = time.clock()
        for i in range(0, len(train_set), args.batch_size):
            if (i + args.batch_size + 1) > len(train_set):
                batch_uvls = train_set[i:]
            else:
                batch_uvls = train_set[i: i + args.batch_size]

            pairs = [[uvl[0], uvl[1]] for uvl in batch_uvls]
            labels = t.tensor([uvl[2] for uvl in batch_uvls]).view(-1).float()
            if t.cuda.is_available():
                labels = labels.to(args.device)
            predicts = model(pairs, embedding_matrix)

            loss = criterion(predicts, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_auc, train_acc = model.ctr_eval(train_set, embedding_matrix, args.batch_size)
        eval_auc, eval_acc = model.ctr_eval(eval_set, embedding_matrix, args.batch_size)
        test_auc, test_acc = model.ctr_eval(test_set, embedding_matrix, args.batch_size)
        eval_auc_list.append(eval_auc)
        test_auc_list.append(test_auc)
        test_acc_list.append(test_acc)
        models.append(copy.deepcopy(model))
        end = time.clock()

        print('epoch: %d' % (epoch + 1), end='\t')
        print('train_auc: %.4f, train_acc: %.4f' % (train_auc, train_acc), end='\t')
        print('eval_auc: %.4f, eval_acc: %.4f' % (eval_auc, eval_acc), end='\t')
        print('test_auc: %.4f, test_acc: %.4f' % (test_auc, test_acc), end='\t')
        print('time: %d' % round(end - start))

    max_eval_auc = max(eval_auc_list)
    n_epochs = eval_auc_list.index(max_eval_auc) + 1
    print(args.dataset, end='\t')
    print('dim: %d' % args.dim, end='\t')
    print('lr: %1.0e' % args.lr, end='\t')
    print('l2: %1.0e' % args.l2, end='\t')
    print('batch_size: %d' % args.batch_size, end='\t')
    print('n_epoch: %d' % n_epochs, end='\t')
    print('max eval_auc: %.4f' % max_eval_auc, end='\t')
    print('test_auc: %.4f, test_acc: %.4f' % (
        test_auc_list[n_epochs - 1], test_acc_list[n_epochs - 1]))

    test_auc = 0
    test_acc = 0
    recall_list = []
    ndcg_list = []
    if is_topk:
        optim_model = models[n_epochs - 1]
        test_auc, test_acc = optim_model.ctr_eval(test_set, embedding_matrix, args.batch_size)
        scores = optim_model.get_scores(rec, embedding_matrix)
        print(args.dataset + '\t test auc: %.4f acc: %.4f' % (test_auc, test_acc))

        for K in [1, 2, 5, 10, 20, 50, 100]:
            recall, ndcg = optim_model.topk_eval(scores, test_records, K)
            recall_list.append(recall)
            ndcg_list.append(ndcg)

        print('Recall@K: ', recall_list)
        print('NDCG@K: ', ndcg_list)

    return max_eval_auc, test_auc, test_acc, recall_list, ndcg_list

