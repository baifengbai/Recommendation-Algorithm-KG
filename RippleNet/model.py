import math
import time

import numpy as np
import torch.nn as nn
import torch as t
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score


class RippleNet(nn.Module):

    def __init__(self, dim, n_entities, H, n_rel, l1, l2):
        super(RippleNet, self).__init__()
        # t.manual_seed(255)
        # t.cuda.manual_seed(255)
        self.dim = dim
        self.H = H
        self.l1 = l1
        self.l2 = l2
        self.ent_emb = nn.Parameter(t.randn(n_entities, dim))
        self.rel_emb = nn.Parameter(t.randn(n_rel, dim, dim))
        self.criterion = nn.BCELoss(reduction='sum')

    def forward(self, pairs, ripple_sets):

        users = [pair[0] for pair in pairs]
        items = [pair[1] for pair in pairs]
        item_embeddings = self.ent_emb[items]
        heads_list, relations_list, tails_list = self.get_head_relation_and_tail(users, ripple_sets)
        user_represents = self.get_vector(items, heads_list, relations_list, tails_list)

        predicts = t.sigmoid((user_represents * item_embeddings).sum(dim=1))

        return predicts

    def get_head_relation_and_tail(self, users, ripple_sets):

        heads_list = []
        relations_list = []
        tails_list = []
        for h in range(self.H):
            l_head_list = []
            l_relation_list = []
            l_tail_list = []

            for user in users:

                l_head_list.extend(ripple_sets[user][h][0])
                l_relation_list.extend(ripple_sets[user][h][1])
                l_tail_list.extend(ripple_sets[user][h][2])

            heads_list.append(l_head_list)
            relations_list.append(l_relation_list)
            tails_list.append(l_tail_list)

        return heads_list, relations_list, tails_list

    def get_vector(self, items, heads_list, relations_list, tails_list):

        o_list = []
        item_embeddings = self.ent_emb[items].view(-1, self.dim, 1)
        for h in range(self.H):
            head_embeddings = self.ent_emb[heads_list[h]].view(len(items), -1, self.dim, 1)
            relation_embeddings = self.rel_emb[relations_list[h]].view(len(items), -1, self.dim, self.dim)
            tail_embeddings = self.ent_emb[tails_list[h]].view(len(items), -1, self.dim)

            Rh = t.matmul(relation_embeddings, head_embeddings).view(len(items), -1, self.dim)
            hRv = t.matmul(Rh, item_embeddings)
            pi = t.softmax(hRv, dim=1)
            o_embeddings = (pi * tail_embeddings).sum(dim=1)
            o_list.append(o_embeddings)

        return sum(o_list)

    def get_scores(self, rec, ripple_sets):
        scores = {}
        self.eval()
        for user in (rec):

            items = list(rec[user])
            pairs = [[user, item] for item in items]
            predict = self.forward(pairs, ripple_sets).cpu().view(-1).detach().numpy().tolist()
            # print(predict)
            n = len(pairs)
            user_scores = {items[i]: predict[i] for i in range(n)}
            user_list = list(dict(sorted(user_scores.items(), key=lambda x: x[1], reverse=True)).keys())
            scores[user] = user_list
        self.train()
        # print('=========================')
        return scores

    def ctr_eval(self, data, ripple_sets, batch_size):
        self.eval()
        pred_label = []
        for i in range(0, len(data), batch_size):
            if (i + batch_size + 1) > len(data):
                batch_uvls = data[i:]
            else:
                batch_uvls = data[i: i + batch_size]

            pairs = [[uvl[0], uvl[1]] for uvl in batch_uvls]

            predicts = self.forward(pairs, ripple_sets).view(-1).cpu().detach().numpy().tolist()
            pred_label.extend(predicts)

        self.train()
        true_label = [line[2] for line in data]
        auc = roc_auc_score(true_label, pred_label)
        np_array = np.array(pred_label)
        np_array[np_array >= 0.5] = 1
        np_array[np_array < 0.5] = 0
        # print(np_array.tolist()[:10])
        # print(true_label[:10])
        acc = accuracy_score(true_label, np_array.tolist())

        return auc, acc

    def computer_loss(self, labels, predicts, users, ripple_sets):

        base_loss = self.criterion(predicts, labels)
        kg_loss = 0
        l2_loss = 0
        for h in range(self.H):
            h_head_list = []
            h_relation_list = []
            h_tail_list = []
            for user in users:
                h_head_list.extend(ripple_sets[user][h][0])
                h_relation_list.extend(ripple_sets[user][h][1])
                h_tail_list.extend(ripple_sets[user][h][2])

            head_emb = self.ent_emb[h_head_list].view(-1, 1, self.dim)  # (n, dim)-->(n, 1, dim)
            rel_emb = self.rel_emb[h_relation_list].view(-1, self.dim, self.dim)  # (n, dim, dim)
            tail_emb = self.ent_emb[h_relation_list].view(-1, self.dim, 1)  # (n, dim)-->(n, dim, 1)

            Rt = t.matmul(rel_emb, tail_emb)  # (n, dim, 1)
            hRt = t.matmul(head_emb, Rt)  # (n, 1, 1)

            kg_loss = kg_loss + t.sigmoid(hRt).mean()

            l2_loss = l2_loss + ((head_emb * head_emb).sum(dim=[1, 2])).mean()
            l2_loss = l2_loss + ((rel_emb * rel_emb).sum(dim=[1, 2])).mean()
            l2_loss = l2_loss + ((tail_emb * tail_emb).sum(dim=[1, 2])).mean()

        return base_loss + (self.l1 / 2) * kg_loss + (self.l2 / 2) * l2_loss

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


