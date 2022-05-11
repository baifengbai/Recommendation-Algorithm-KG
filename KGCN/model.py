import math

import numpy as np
import torch as t
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score


class KGCN(nn.Module):

    def __init__(self, n_entity, n_relation, dim, n_iter, n_neighbors):
        super(KGCN, self).__init__()
        # t.manual_seed(255)
        # t.cuda.manual_seed(255)

        self.entity_emb_matrix = nn.Parameter(t.randn(n_entity, dim))
        self.relation_emb_matrix = nn.Parameter(t.randn(n_relation, dim))
        self.W_sum = nn.Linear(dim, dim)
        self.dim = dim
        self.n_iter = n_iter
        self.n_neighbors = n_neighbors

    def forward(self, pairs, adj_entity_np, adj_relation_np):

        users = [pair[0] for pair in pairs]
        items = [pair[1] for pair in pairs]
        entities, relations = self.get_neighbors(items, adj_entity_np, adj_relation_np)

        user_embeddings = self.entity_emb_matrix[users]
        entity_vectors = [self.entity_emb_matrix[entities[i]].reshape(len(pairs), -1, self.dim) for i in range(len(entities))]
        relation_vectors = [self.relation_emb_matrix[relations[i]].reshape(len(pairs), -1, self.dim) for i in range(len(relations))]
        # arr = np.random.randint(1, 3, (1024, 5))
        # print(self.relation_emb_matrix[relations[0]].shape, relations[0].shape)
        # print(self.relation_emb_matrix[arr].shape, arr.shape)
        # print(self.entity_emb_matrix[entities[1]].shape, entities[1].shape, type(entities[1]))

        for i in range(self.n_iter):

            entity_vectors = self.gcn_layer(user_embeddings, entity_vectors, relation_vectors, i)

        item_embeddings = entity_vectors[0].reshape(-1, self.dim)

        predict = (user_embeddings * item_embeddings).sum(dim=1)

        return t.sigmoid(predict)

    def gcn_layer(self, user_embeddings, entity_vectors, relation_vectors, l):

        n = user_embeddings.shape[0]
        next_entity_vectors = []
        shape = [n, -1, self.n_neighbors, self.dim]
        for i in range(self.n_iter - l):
            # print(relation_vectors[i].shape, entity_vectors[i + 1].shape)
            relation_scores = (user_embeddings.view(n, 1, 1, self.dim) * relation_vectors[i].view(shape)).sum(dim=-1)
            relation_scores = relation_scores.reshape(n, -1, self.n_neighbors, 1)
            normalize_relation_scores = t.softmax(relation_scores, dim=-2)
            neighbor_embeddings = normalize_relation_scores * entity_vectors[i + 1].reshape(shape)
            neighbor_embeddings = neighbor_embeddings.sum(dim=-2)
            entity_and_neighbors = entity_vectors[i].view(n, -1, self.dim) + neighbor_embeddings
            if l == self.n_iter-1:
                next_entity_vectors.append(t.tanh(self.W_sum(entity_and_neighbors)))
            else:
                next_entity_vectors.append(t.relu(self.W_sum(entity_and_neighbors)))

        return next_entity_vectors

    def get_neighbors(self, seeds, adj_entity_np, adj_relation_np):

        entities = [seeds]
        relations = []
        n = len(seeds)
        for i in range(self.n_iter):
            entities.append(adj_entity_np[entities[i]].reshape(-1))
            relations.append(adj_relation_np[entities[i]].reshape(-1))

        return entities, relations

    def get_scores(self, rec, adj_entity_np, adj_relation_np):
        scores = {}
        self.eval()
        for user in (rec):
            items = list(rec[user])
            pairs = [[user, item] for item in items]
            predict = self.forward(pairs, adj_entity_np, adj_relation_np).cpu().view(-1).detach().numpy().tolist()
            n = len(pairs)
            user_scores = {items[i]: predict[i] for i in range(n)}
            user_list = list(dict(sorted(user_scores.items(), key=lambda x: x[1], reverse=True)).keys())
            scores[user] = user_list
        self.train()
        return scores

    def ctr_eval(self, pairs, adj_entity_np, adj_relation_np, batch_size):
        self.eval()
        true_labels = [i[2] for i in pairs]
        pred = []
        for i in range(0, len(pairs), batch_size):
            next_i = min([len(pairs), i + batch_size])
            predict = self.forward(pairs[i: next_i], adj_entity_np, adj_relation_np).cpu().detach().numpy()
            pred.extend(predict.tolist())
        self.train()

        pred = np.array(pred)
        auc = roc_auc_score(true_labels, pred)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        acc = accuracy_score(true_labels, pred)

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

        return Recall, NDCG


