import time

import numpy as np

from data_loader import load_kg, get_hrtts, get_uvvs, get_records, load_data
from model import CKE
import torch as t
import torch.optim as optim
import math
import copy


def train(args, is_topk):
    data = load_data(args.dataset)
    n_entity, n_relation = data[0], data[1]
    train_set, eval_set, test_set = data[2], data[3], data[4]
    rec, kg_dict = data[5], data[6]
    hrtts = get_hrtts(kg_dict)
    test_records = get_records(test_set)
    model = CKE(n_entity, n_relation, args.dim,  args.lu, args.lv, args.lM, args.lr, args.lI)

    if t.cuda.is_available():
        model = model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2)

    train_data = [hrtts, get_uvvs(train_set)]
    print(args.dataset + '----------------------------')
    print('dim: %d' % args.dim, end='\t')
    print('learning_rate: %1.0e' % args.learning_rate, end='\t')
    print('l2: %1.0e' % args.l2, end='\t')
    print('batch_size: %d' % args.batch_size)

    eval_auc_list = []
    test_auc_list = []
    test_acc_list = []
    models = []
    for epoch in range(args.epochs):

        start = time.clock()
        size = len(train_data[0])
        start_index = 0
        while start_index < size:
            if start_index + args.batch_size <= size:
                hrtts = train_data[0][start_index: start_index + args.batch_size]
            else:
                hrtts = train_data[0][start_index:]
            loss = -model(hrtts, 'kg')
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            start_index += args.batch_size

        start_index = 0
        size = len(train_data[-1])
        while start_index < size:
            if start_index + args.batch_size <= size:
                pairs = train_data[-1][start_index: start_index + args.batch_size]
            else:
                pairs = train_data[-1][start_index:]
            loss = -model(pairs, 'cf')
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            start_index += args.batch_size

        train_auc, train_acc = model.ctr_eval(train_set, args.batch_size)
        eval_auc, eval_acc = model.ctr_eval(eval_set, args.batch_size)
        test_auc, test_acc = model.ctr_eval(test_set, args.batch_size)
        eval_auc_list.append(eval_auc)
        test_auc_list.append(test_auc)
        test_acc_list.append(test_acc)
        models.append(copy.deepcopy(model))
        end = time.clock()
        print('epoch %d \t train auc: %.4f acc: %.4f \t eval auc: %.4f acc:%.4f \t test auc: %.4f acc: %.4f \t time: %d'
              % ((epoch + 1), train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc, (end - start)))
    max_eval_auc = max(eval_auc_list)
    n_epochs = eval_auc_list.index(max_eval_auc) + 1
    print(args.dataset, end='\t')
    print('dim: %d' % args.dim, end='\t')
    print('learning_rate: %1.0e' % args.learning_rate, end='\t')
    print('l2: %1.0e' % args.l2, end='\t')
    print('batch_size: %d' % args.batch_size, end='\t')
    print('n_epoch: %d' % n_epochs, end='\t')
    print('max eval auc: %.4f' % max_eval_auc, end='\t')
    print('test auc: %.4f, acc: %.4f' % (test_auc_list[n_epochs - 1], test_acc_list[n_epochs - 1]))
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
