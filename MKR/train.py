import numpy as np
import torch as t
from tqdm import tqdm

from data_loader import get_hrtts, get_records
import torch.optim as optim
from model import MKR
import time
import math
from data_loader import load_data
import copy


def train(args, is_topk):
    data = load_data(args.dataset)
    n_entity, n_relation = data[0], data[1]
    train_set, eval_set, test_set = data[2], data[3], data[4]
    rec, kg_dict = data[5], data[6]
    hrtts = get_hrtts(kg_dict)
    test_records = get_records(test_set)
    model = MKR(args.dim, args.L, args.T, args.l1, n_entity, n_relation)
    if t.cuda.is_available():
        model = model.to(args.device)
    print(args.dataset + '-----------------------------------')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    print('dim: %d' % args.dim, end='\t')
    print('L: %d' % args.L, end='\t')
    print('l1: %1.0e' % args.l1, end='\t')
    print('lr: %1.0e' % args.lr, end='\t')
    print('l2: %1.0e' % args.l2, end='\t')
    print('batch_size: %d' % args.batch_size)
    eval_auc_list = []
    test_auc_list = []
    test_acc_list = []
    models = []
    for epoch in (range(args.epochs)):
        start = time.clock()
        model.train()
        size = len(train_set)
        for j in range(model.T):
            for i in range(0, size, args.batch_size):
                next_i = min([size, i+args.batch_size])
                data = train_set[i: next_i]
                loss = model.cal_rs_loss(data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        size = len(hrtts)
        for i in range(0, size, args.batch_size):
            next_i = min([size, i + args.batch_size])
            data = hrtts[i: next_i]
            loss = model.cal_kg_loss(data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_auc, train_acc = model.ctr_eval(train_set, args.batch_size)
        eval_auc, eval_acc = model.ctr_eval(eval_set, args.batch_size)
        test_auc, test_acc = model.ctr_eval(test_set, args.batch_size)
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
    print('L: %d' % args.L, end='\t')
    print('l1: %1.0e' % args.l1, end='\t')
    print('lr: %1.0e' % args.lr, end='\t')
    print('l2: %1.0e' % args.l2, end='\t')
    print('batch_size: %d' % args.batch_size, end='\t')
    print('n_epoch: %d' % n_epochs, end='\t')
    print('max eval_auc: %.4f' % max_eval_auc, end='\t')
    print('test_auc: %.4f, test_acc: %.4f' % (test_auc_list[n_epochs - 1], test_acc_list[n_epochs - 1]))

    test_auc = 0
    test_acc = 0
    recall_list = []
    ndcg_list = []
    if is_topk:
        optim_model = models[n_epochs - 1]
        test_auc, test_acc = optim_model.ctr_eval(test_set, args.batch_size)
        scores = optim_model.get_scores(rec)
        print(args.dataset + '\t test auc: %.4f acc: %.4f' % (test_auc, test_acc))

        for K in [1, 2, 5, 10, 20, 50, 100]:
            recall, ndcg = optim_model.topk_eval(scores, test_records, K)
            recall_list.append(recall)
            ndcg_list.append(ndcg)

        print('Recall@K: ', recall_list)
        print('NDCG@K: ', ndcg_list)

    return max_eval_auc, test_auc, test_acc, recall_list, ndcg_list


