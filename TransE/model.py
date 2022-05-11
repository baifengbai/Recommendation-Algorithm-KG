import torch.nn as nn
import torch as t
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random


class TransE(nn.Module):

    def __init__(self, dim, n_entity, n_relation):
        super(TransE, self).__init__()
        t.manual_seed(255)
        t.cuda.manual_seed(255)
        self.entity_embedding_matrix = nn.Parameter(t.randn(n_entity, dim))
        self.relation_embedding_matrix = nn.Parameter(t.randn(n_relation, dim))

    def forward(self, data):
        head_embeddings = self.entity_embedding_matrix[[i[0] for i in data]]
        relation_embeddings = self.relation_embedding_matrix[[i[1] for i in data]]
        pos_tail_embeddings = self.entity_embedding_matrix[[i[2] for i in data]]
        neg_tail_embeddings = self.entity_embedding_matrix[[i[3] for i in data]]

        pos_scores = (head_embeddings + relation_embeddings - pos_tail_embeddings).norm(p=2, dim=1)
        neg_scores = (head_embeddings + relation_embeddings - neg_tail_embeddings).norm(p=2, dim=1)
        pos_scores = pos_scores * pos_scores
        neg_scores = neg_scores * neg_scores

        loss = -(pos_scores.sum() - neg_scores.sum())

        return loss

    def get_score(self, heads, relations, tails):
        head_embeddings = self.entity_embedding_matrix[heads]
        relation_embeddings = self.relation_embedding_matrix[relations]
        tail_embeddings = self.entity_embedding_matrix[tails]

        scores = (head_embeddings + relation_embeddings - tail_embeddings).norm(p=2, dim=1)
        return scores.cpu().view(-1).detach().numpy().tolist()

    def get_relation_embedding(self, relations):
        return self.relation_embedding_matrix[relations]

    def get_entity_embedding(self, entities):
        return self.entity_embedding_matrix[entities]

    def get_predict(self, pairs, relation):
        head_embeddings = self.entity_embedding_matrix[[pair[0] for pair in pairs]]
        relation_embeddings = self.relation_embedding_matrix[[relation]]
        tail_embeddings = self.entity_embedding_matrix[[pair[1] for pair in pairs]]
        hr = head_embeddings + relation_embeddings
        predict = (hr * tail_embeddings).sum(dim=1)
        return t.sigmoid(predict)








