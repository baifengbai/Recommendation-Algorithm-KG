import math

import torch as t
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import time
import networkx as nx
from tqdm import tqdm


class Dnn(nn.Module):

    def __init__(self, int_dim, dim):
        # t.manual_seed(255)
        # t.cuda.manual_seed(255)
        super(Dnn, self).__init__()

        self.l1 = nn.Linear(int_dim, 1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, 256)

    def forward(self, x):

        y = self.l1(x)
        y = t.relu(y)
        y = self.l2(y)
        y = t.relu(y)
        y = self.l3(y)
        y = t.relu(y)

        return y


class WideDeep(nn.Module):

    def __init__(self, int_dim, dim):
        # t.manual_seed(255)
        # t.cuda.manual_seed(255)
        super(WideDeep, self).__init__()
        self.dnn = Dnn(int_dim, dim)
        self.final_liner = nn.Linear(256, 1)
        self.wide_liner = nn.Linear(2*dim, 1)
        self.dim = dim

    def forward(self, pairs, embedding_matrix):

        user_vectors = embedding_matrix[[pair[0] for pair in pairs]]
        item_vectors = embedding_matrix[[pair[1] for pair in pairs]]
        deep_x = t.cat([user_vectors, item_vectors], dim=1)
        wide_x = t.cat([user_vectors, item_vectors], dim=1)
        wide_y = self.wide_liner(wide_x)
        wide_y = t.relu(wide_y)

        deep_y = self.dnn(deep_x)
        deep_y = self.final_liner(deep_y)
        deep_y = t.relu(deep_y)

        predict = t.sigmoid(deep_y + wide_y).view(-1)

        return predict

    def get_scores(self, rec, embedding_matrix):

        scores = {}

        for u in (rec):
            pairs = [[u, item] for item in rec[u]]
            predict_np = self.forward(pairs, embedding_matrix).cpu().detach().numpy()
            n = predict_np.shape[0]
            user_scores = {rec[u][i]: predict_np[i] for i in range(n)}
            user_list = list(dict(sorted(user_scores.items(), key=lambda x: x[1], reverse=True)).keys())
            scores[u] = user_list

        return scores

    def ctr_eval(self, data, embedding_matrix, batch_size):
        self.eval()
        true_labels = [i[2] for i in data]
        pred_labels = []
        for i in range(0, len(data), batch_size):
            next_i = min([i + batch_size, len(data)])
            predict = self.forward(data[i: next_i], embedding_matrix).cpu().detach().numpy()
            pred_labels.extend(predict.tolist())
        self.train()

        pred = np.array(pred_labels)
        auc = roc_auc_score(true_labels, pred)
        # auc = round(auc, 4)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        acc = accuracy_score(true_labels, pred)
        # acc = round(acc, 4)

        return auc, acc

    def topk_eval(self, scores, test_records, K):

        recall_sum = 0
        ndcg_sum = 0
        for user in scores:
            rank_items = scores[user][:K]
            hit_num = len(set(rank_items) & set(test_records[user]))
            recall = hit_num / len(test_records[user])
            n = len(rank_items)
            a = sum([1 / math.log(i+2, 2) for i in range(n) if rank_items[i] in test_records[user]])
            b = sum([1 / math.log(i+2, 2) for i in range(len(test_records[user]))])
            ndcg = a / b

            recall_sum += recall
            ndcg_sum += ndcg

        Recall = recall_sum / len(scores)
        NDCG = ndcg_sum / len(scores)

        return round(Recall, 4), round(NDCG, 4)







