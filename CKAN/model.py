import math

import numpy as np
import torch as t
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score


class DNN(nn.Module):

    def __init__(self, dim):
        # t.manual_seed(255)
        # t.cuda.manual_seed(255)
        super(DNN, self).__init__()
        self.l1 = nn.Linear(2*dim, dim)
        self.l2 = nn.Linear(dim, dim)
        self.l3 = nn.Linear(dim, 1)

    def forward(self, x):
        y = self.l1(x)
        y = t.relu(y)
        y = self.l2(y)
        y = t.relu(y)
        y = self.l3(y)
        y = t.sigmoid(y)

        return y


class CKAN(nn.Module):

    def __init__(self, n_entities, dim, n_relations, L, agg):
        super(CKAN, self).__init__()
        # t.manual_seed(255)
        # t.cuda.manual_seed(255)
        entity_embedding_matrix = t.randn(n_entities, dim)
        nn.init.xavier_normal_(entity_embedding_matrix)
        self.entity_embedding_matrix = nn.Parameter(entity_embedding_matrix)
        self.rec_embedding_matrix = nn.Parameter(entity_embedding_matrix)
        self.dim = dim
        rel_embeeding_matrix = t.randn(n_relations, dim)
        nn.init.xavier_normal_(rel_embeeding_matrix)
        self.rel_embedding_matrix = nn.Parameter(rel_embeeding_matrix)
        self.L = L
        self.dnn = DNN(dim)
        self.agg = agg
        self.criterion = nn.BCELoss()
        if agg == 'concat':
            self.Wagg = nn.Linear((L+1)*dim, dim)
        else:
            self.Wagg = nn.Linear(dim, dim)

    def forward(self, pairs, ripple_sets):
        users = [pair[0] for pair in pairs]
        items = [pair[1] for pair in pairs]
        u_heads_list, u_relations_list, u_tails_list, u_entity_list = self.get_head_relation_and_tail(users, ripple_sets)
        v_heads_list, v_relations_list, v_tails_list, v_entity_list = self.get_head_relation_and_tail(items,
                                                                                                      ripple_sets)
        users_embeddings = self.get_item_embedding(u_heads_list, u_relations_list, u_tails_list, u_entity_list, 'user', len(users))
        items_embeddings = self.get_item_embedding(v_heads_list, v_relations_list, v_tails_list, v_entity_list, 'item', len(items))
        predicts = t.sigmoid((users_embeddings * items_embeddings).sum(dim=1)).view(-1)

        return predicts.view(-1)

    def get_head_relation_and_tail(self, os, ripple_sets):

        heads_list = []
        relations_list = []
        tails_list = []
        entity_list = []
        for l in range(self.L+1):
            l_head_list = []
            l_relation_list = []
            l_tail_list = []

            for o in os:
                if l == 0:
                    entity_list.extend(ripple_sets[o][0])
                else:
                    l_head_list.extend(ripple_sets[o][l][0])
                    l_relation_list.extend(ripple_sets[o][l][1])
                    l_tail_list.extend(ripple_sets[o][l][2])

            heads_list.append(l_head_list)
            relations_list.append(l_relation_list)
            tails_list.append(l_tail_list)

        return heads_list, relations_list, tails_list, entity_list

    def get_item_embedding(self, heads_list, relations_list, tails_list, entity_list, o, n):
        e_list = []
        if o == 'user':
            e_list.append(self.rec_embedding_matrix[entity_list].view(n, -1, self.dim).mean(dim=1))
        else:
            e_list.append(self.entity_embedding_matrix[entity_list].view(n, -1, self.dim).mean(dim=1))

        for l in range(1, self.L+1):
            head_embeddings = self.entity_embedding_matrix[heads_list[l]]
            relation_embeddings = self.rel_embedding_matrix[relations_list[l]]
            tail_embeddings = self.entity_embedding_matrix[tails_list[l]]

            pi = self.dnn(t.cat([head_embeddings, relation_embeddings], dim=1))
            pi = t.softmax(pi.view(n, -1, 1), dim=1)
            a = (pi * tail_embeddings.view(n, -1, self.dim)).sum(dim=1)
            e_list.append(a)

        return self.aggregator(e_list)

    def aggregator(self, e_list):
        # print(len(e_list))
        embedding = t.cat(e_list, dim=1)
        # print(embedding.shape, self.Wagg.weight.shape)
        if self.agg == 'concat':
            return t.sigmoid(self.Wagg(embedding))
        elif self.agg == 'sum':
            return t.sigmoid(self.Wagg(embedding.sum(dim=0).view(1, self.dim)))
        else:
            return t.sigmoid(self.Wagg(embedding.max(dim=0)[0].view(1, self.dim)))

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
        acc = accuracy_score(true_label, np_array.tolist())

        return auc, acc

    def topk_eval(self, scores, test_records, K):
        recall_sum = 0
        ndcg_sum = 0
        for user in scores:
            rank_items = scores[user][:K]
            hit_num = len(set(rank_items) & set(test_records[user]))
            recall = hit_num / len(test_records[user])
            n = len(rank_items)
            a = sum([1 / math.log(i + 2, 2) for i in range(n) if rank_items[i] in test_records[user]])
            b = sum([1 / math.log(i + 2, 2) for i in range(len(test_records[user]))])
            ndcg = a / b

            recall_sum += recall
            ndcg_sum += ndcg

        Recall = recall_sum / len(scores)
        NDCG = ndcg_sum / len(scores)

        return round(Recall, 4), round(NDCG, 4)

