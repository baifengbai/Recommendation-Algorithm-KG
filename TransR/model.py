import torch.nn as nn
import torch as t
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random


class TransR(nn.Module):

    def __init__(self, n_ent, n_rel, dim):
        t.manual_seed(255)
        t.cuda.manual_seed(255)
        super(TransR, self).__init__()
        self.ent_emb = nn.Parameter(t.randn(n_ent, dim))
        self.Mr_emb = nn.Parameter(t.randn(n_rel, dim, dim))
        self.rel_emb = nn.Parameter(t.randn(n_rel, dim))
        self.dim = dim

    def forward(self, hs, rs, pos_ts, neg_ts):
        head_ids = [i for i in hs]
        pos_ids = [i for i in pos_ts]
        neg_ids = [i for i in neg_ts]
        h_emb = self.ent_emb[head_ids].view(-1, 1, self.dim)  # (n, dim)-->(n, 1, dim)
        Mr_vectors = self.Mr_emb[rs]  # (n, dim, dim)
        r_emb = self.rel_emb[rs].view(-1, 1, self.dim)  # (n, dim)-->(n, 1, dim)
        pos_t_emb = self.ent_emb[pos_ids].view(-1, 1, self.dim)  # (n, dim)-->(n, 1, dim)
        # print(neg_ts)
        neg_t_emb = self.ent_emb[neg_ids].view(-1, 1, self.dim)  # (n, dim)-->(n, 1, dim)

        h_emb_nor = F.normalize(h_emb, dim=(1, 2))
        Mr_vectors_nor = F.normalize(Mr_vectors, dim=(1, 2))
        r_emb_nor = F.normalize(r_emb, dim=(1, 2))
        pos_t_emb_nor = F.normalize(pos_t_emb, dim=(1, 2))
        neg_t_emb_nor = F.normalize(neg_t_emb, dim=(1, 2))

        hr = t.matmul(h_emb_nor, Mr_vectors_nor)  # (n, 1, dim)
        pos_tr = t.matmul(pos_t_emb_nor, Mr_vectors_nor)  # (n, 1, dim)
        neg_tr = t.matmul(neg_t_emb_nor, Mr_vectors_nor)  # (n, 1, dim)

        return t.norm(hr + r_emb_nor - pos_tr, dim=(1, 2)) ** 2, t.norm(hr + r_emb_nor - neg_tr, dim=(1, 2)) ** 2








