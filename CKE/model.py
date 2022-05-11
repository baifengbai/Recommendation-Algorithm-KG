import time

import torch as t
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import math


class CKE(nn.Module):

    def __init__(self, n_entities, n_rels, dim,  lu, lv, lM, lr, lI):
        # t.manual_seed(128)
        # t.cuda.manual_seed(128)
        # np.random.seed(255)

        super(CKE, self).__init__()
        self.dim = dim
        self.lu = lu
        self.lv = lv
        self.lM = lM
        self.lr = lr
        self.lI = lI
        user_emb_matrix = t.randn(n_entities, dim)
        item_emb_matrix = t.randn(n_entities, dim)
        ent_emb_matrix = t.randn(n_entities, dim)
        Mr_matrix = t.randn(n_rels, dim, dim)
        rel_emb_matrix = t.randn(n_rels, dim)
        nn.init.xavier_uniform_(user_emb_matrix)
        nn.init.xavier_uniform_(item_emb_matrix)
        nn.init.xavier_uniform_(ent_emb_matrix)
        nn.init.xavier_uniform_(Mr_matrix)
        nn.init.xavier_uniform_(rel_emb_matrix)
        self.user_emb_matrix = nn.Parameter(user_emb_matrix)
        self.item_emb_matrix = nn.Parameter(item_emb_matrix)
        self.ent_emb_matrix = nn.Parameter(ent_emb_matrix)
        self.Mr_matrix = nn.Parameter(Mr_matrix)
        self.rel_emb_matrix = nn.Parameter(rel_emb_matrix)
        # self.user_emb_matrix = nn.Parameter(t.from_numpy(np.random.normal(loc=0, scale=1e-5, size=[n_entities, dim])))
        # # print(self.user_emb_matrix.shape)
        # scale = 1e-5  # job: 1e-5
        # self.item_emb_matrix = nn.Parameter(t.from_numpy(np.random.normal(loc=0, scale=scale, size=[n_entities, dim])))
        # self.ent_emb_matrix = nn.Parameter(t.from_numpy(np.random.normal(loc=0, scale=scale, size=[n_entities, dim])))
        # self.Mr_matrix = nn.Parameter(t.from_numpy(np.random.normal(loc=0, scale=scale, size=[n_rels, dim, dim])))
        # self.rel_emb_matrix = nn.Parameter(t.from_numpy(np.random.normal(loc=0, scale=scale, size=[n_rels, dim])))

    def forward(self, data, name):
        if name == 'kg':
            # print(data)
            heads_id = [i[0] for i in data]
            relations_id = [i[1] for i in data]
            pos_tails_id = [i[2] for i in data]
            neg_tails_id = [i[3] for i in data]
            head_emb = self.ent_emb_matrix[heads_id].view(-1, 1, self.dim)
            rel_emb = self.rel_emb_matrix[relations_id].view(-1, 1, self.dim)
            pos_tail_emb = self.ent_emb_matrix[pos_tails_id].view(-1, 1, self.dim)
            neg_tail_emb = self.ent_emb_matrix[neg_tails_id].view(-1, 1, self.dim)
            Mr = self.Mr_matrix[relations_id]

            pos_stru_scores = (t.matmul(head_emb, Mr) + rel_emb - t.matmul(pos_tail_emb, Mr)).norm(dim=[1, 2]) ** 2
            neg_stru_scores = (t.matmul(head_emb, Mr) + rel_emb - t.matmul(neg_tail_emb, Mr)).norm(dim=[1, 2]) ** 2
            # print(t.log(t.sigmoid(pos_stru_scores - neg_stru_scores)))
            stru_loss = t.sigmoid(pos_stru_scores - neg_stru_scores)
            stru_loss = t.log(stru_loss).sum()
            return stru_loss
        else:

        # print(uvv)
            users_id = [i[0] for i in data]
            poss_id = [i[1] for i in data]
            negs_id = [i[2] for i in data]
            users_emb = self.user_emb_matrix[users_id]
            pos_items_emb = self.item_emb_matrix[poss_id] + self.ent_emb_matrix[poss_id]
            neg_items_emb = self.item_emb_matrix[negs_id] + self.ent_emb_matrix[negs_id]
            base_loss = t.sigmoid(t.mul(users_emb, pos_items_emb).sum(dim=1) - t.mul(users_emb, neg_items_emb).sum(dim=1))
            base_loss = t.log(base_loss).sum()

            return base_loss

    def get_scores(self, rec):
        scores = {}
        self.eval()
        for user in (rec):
            pairs = [[user, item] for item in rec[user]]
            predict = self.get_predict(pairs)
            # print(predict)
            n = len(pairs)
            user_scores = {rec[user][i]: predict[i] for i in range(n)}
            user_list = list(dict(sorted(user_scores.items(), key=lambda x: x[1], reverse=True)).keys())
            scores[user] = user_list
        self.train()
        return scores

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

        return Recall, NDCG

    def get_predict(self, pairs):

        users = [pair[0] for pair in pairs]
        items = [pair[1] for pair in pairs]
        user_emb = self.user_emb_matrix[users]

        item_emb = self.item_emb_matrix[items] + self.ent_emb_matrix[items]
        score = (user_emb * item_emb).sum(dim=1)

        return score.cpu().detach().view(-1).numpy().tolist()

    def ctr_eval(self, data, batch_size):
        self.eval()
        true_labels = [i[2] for i in data]
        pred_labels = []
        for i in range(0, len(data), batch_size):
            next_i = min([i + batch_size, len(data)])
            predict = self.get_predict(data[i: next_i])
            pred_labels.extend(predict)
        self.train()

        pred = np.array(pred_labels)
        auc = roc_auc_score(true_labels, pred)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        acc = accuracy_score(true_labels, pred)

        return auc, acc


