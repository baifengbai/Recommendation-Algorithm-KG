import math

import numpy as np
import torch as t
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score


class CrossAndCompressUnit(nn.Module):
    def __init__(self, dim):
        super(CrossAndCompressUnit, self).__init__()
        # t.manual_seed(255)
        # t.cuda.manual_seed(255)
        self.dim = dim
        self.w_vv = nn.Parameter(t.randn(dim, 1))
        self.w_ev = nn.Parameter(t.randn(dim, 1))
        self.w_ve = nn.Parameter(t.randn(dim, 1))
        self.w_vv = nn.Parameter(t.randn(dim, 1))
        self.b_v = nn.Parameter(t.randn(dim, 1))
        self.b_e = nn.Parameter(t.randn(dim, 1))

    def forward(self, v, e):
        C = t.matmul(e.view(-1, self.dim, 1), v)  # (-1, 1, d) * (-1, d, 1) = (-1, d, d)
        v = t.matmul(C, self.w_vv) + t.matmul(C, self.w_ev) + self.b_v
        e = t.matmul(C, self.w_ve) + t.matmul(C, self.w_vv) + self.b_e
        return v.view(-1, 1, self.dim), e.view(-1, 1, self.dim)


class CAC1(nn.Module):
    def __init__(self, dim):
        super(CAC1, self).__init__()
        # t.manual_seed(255)
        # t.cuda.manual_seed(255)
        self.cac1 = CrossAndCompressUnit(dim)

    def forward(self, v, e):
        v, e = self.cac1(v, e)
        return v, e


class CAC2(nn.Module):
    def __init__(self, dim):
        super(CAC2, self).__init__()
        # t.manual_seed(255)
        # t.cuda.manual_seed(255)
        self.cac1 = CrossAndCompressUnit(dim)
        self.cac2 = CrossAndCompressUnit(dim)

    def forward(self, v, e):
        v, e = self.cac1(v, e)
        v, e = self.cac2(v, e)
        return v, e


class CAC3(nn.Module):
    def __init__(self, dim):
        super(CAC3, self).__init__()
        # t.manual_seed(255)
        # t.cuda.manual_seed(255)
        self.cac1 = CrossAndCompressUnit(dim)
        self.cac2 = CrossAndCompressUnit(dim)
        self.cac3 = CrossAndCompressUnit(dim)

    def forward(self, v, e):
        v, e = self.cac1(v, e)
        v, e = self.cac2(v, e)
        v, e = self.cac3(v, e)
        return v, e


class MLP1(nn.Module):
    def __init__(self, int_dim, hidden_dim, out_dim):
        super(MLP1, self).__init__()
        # t.manual_seed(255)
        # t.cuda.manual_seed(255)
        self.l1 = nn.Linear(int_dim, out_dim)

    def forward(self, x):
        y = t.relu(self.l1(x))
        return y


class MLP2(nn.Module):
    def __init__(self, int_dim, hidden_dim, out_dim):
        super(MLP2, self).__init__()
        # t.manual_seed(255)
        # t.cuda.manual_seed(255)
        self.l1 = nn.Linear(int_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        y = t.relu(self.l1(x))
        y = t.relu(self.l2(y))
        return y


class MLP3(nn.Module):
    def __init__(self, int_dim, hidden_dim, out_dim):
        super(MLP3, self).__init__()
        # t.manual_seed(255)
        # t.cuda.manual_seed(255)
        self.l1 = nn.Linear(int_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        y = t.relu(self.l1(x))
        y = t.relu(self.l2(y))
        y = t.relu(self.l3(y))
        return y


class MKR(nn.Module):

    def __init__(self, dim, L, T, l1, n_entities, n_relations):
        super(MKR, self).__init__()
        # t.manual_seed(255)
        # t.cuda.manual_seed(255)
        self.dim = dim
        self.L = L
        self.T = T
        self.l1 = l1
        self.M_k = nn.Linear(2*dim, dim)
        if L == 1:
            self.mlp = MLP1(dim, dim, dim)
            self.cac = CAC1(dim)
        elif L == 2:
            self.mlp = MLP2(dim, dim, dim)
            self.cac = CAC2(dim)
        else:
            self.mlp = MLP3(dim, dim, dim)
            self.cac = CAC3(dim)
        self.rs_entity_embedding = nn.Parameter(t.randn(n_entities, dim))
        self.rs_item_embedding = nn.Parameter(t.randn(n_entities, dim))
        self.e_entity_embedding = nn.Parameter(t.randn(n_entities, dim))
        self.e_item_embedding = nn.Parameter(t.randn(n_entities, dim))
        self.relation_embedding = nn.Parameter(t.randn(n_relations, dim))
        self.criterion = nn.BCELoss()

    def forward(self, data):
        users = self.rs_entity_embedding[[i[0] for i in data]]
        items = self.rs_item_embedding[[i[1] for i in data]].view(-1, 1, self.dim)
        item_entities = self.e_entity_embedding[[i[1] for i in data]].view(-1, 1, self.dim)

        u_L = self.mlp(users)
        v_L = self.cac(items, item_entities)[0].view(-1, self.dim)

        predicts = ((u_L * v_L).sum(dim=1)).view(-1)

        return predicts

    def cal_kg_loss(self, data):

        heads = self.e_entity_embedding[[i[0] for i in data]].view(-1, 1, self.dim)
        relations = self.relation_embedding[[i[1] for i in data]]
        pos_tails = self.e_entity_embedding[[i[2] for i in data]]
        neg_tails = self.e_entity_embedding[[i[3] for i in data]]
        items = self.rs_item_embedding[[i[0] for i in data]].view(-1, 1, self.dim)

        true_scores = self.get_kg_scores(heads, relations, pos_tails, items)
        false_scores = self.get_kg_scores(heads, relations, neg_tails, items)

        return -self.l1 * (true_scores - false_scores)

    def get_kg_scores(self, heads, relations, tails, items):
        h_L = self.cac(items, heads)[1].view(-1, self.dim)
        r_L = self.mlp(relations)

        pred_tails = t.relu(self.M_k(t.cat([h_L, r_L], dim=1)))

        scores = ((pred_tails * tails).sum(dim=1)).sum()

        return scores

    def cal_rs_loss(self, data):
        users = self.rs_entity_embedding[[i[0] for i in data]]
        items = self.rs_item_embedding[[i[1] for i in data]].view(-1, 1, self.dim)
        item_entities = self.e_entity_embedding[[i[1] for i in data]].view(-1, 1, self.dim)
        labels = t.tensor([float(i[2]) for i in data]).view(-1)
        if t.cuda.is_available():
            labels = labels.to(users.device)

        u_L = self.mlp(users)
        v_L = self.cac(items, item_entities)[0].view(-1, self.dim)

        predicts = t.sigmoid(((u_L * v_L).sum(dim=1)).view(-1))
        return self.criterion(predicts, labels)

    def get_scores(self, rec):
        # print('get scores...')
        scores = {}
        self.eval()
        for user in (rec):
            items = list(rec[user])
            pairs = [[user, item] for item in items]
            predict = self.forward(pairs)
            # print(predict)
            n = len(pairs)
            user_scores = {items[i]: predict[i] for i in range(n)}
            user_list = list(dict(sorted(user_scores.items(), key=lambda x: x[1], reverse=True)).keys())
            scores[user] = user_list
        self.train()
        return scores

    def ctr_eval(self, data, batch_size):
        self.eval()
        true_labels = [i[2] for i in data]
        pred_labels = []
        for i in range(0, len(data), batch_size):
            next_i = min([i + batch_size, len(data)])
            predict = self.forward(data[i: next_i]).cpu().detach().numpy()
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

    def topk_eval(self, rec, test_records, K):

        scores = self.get_scores(rec)
        recall_sum = 0
        ndcg_sum = 0
        for user in rec:
            rank_items = scores[user][:K]
            hit_num = len(set(rank_items) & set(test_records[user]))
            recall = hit_num / len(test_records[user])
            n = len(rank_items)
            a = sum([1 / math.log(i+2, 2) for i in range(n) if rank_items[i] in test_records[user]])
            b = sum([1 / math.log(i+2, 2) for i in range(len(test_records[user]))])
            ndcg = a / b

            recall_sum += recall
            ndcg_sum += ndcg

        Recall = recall_sum / len(rec)
        NDCG = ndcg_sum / len(rec)

        return Recall, NDCG






