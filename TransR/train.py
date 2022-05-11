import torch.nn as nn
import torch as t
import torch.nn.functional as F
from model import TransR
from tqdm import tqdm
import random
from data_loader import load_data


def train(args):
    data = load_data(args)
    n_entity = data[0]
    n_relation = data[1]
    kg_dict = data[2]
    hrtts = get_hrtts(kg_dict)
    model = TransR(n_entity, n_relation, args.dim)
    if t.cuda.is_available():
        model = model.to(args.device)
    optimizer = t.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    one = t.tensor([-1])
    criterion = nn.MarginRankingLoss(reduction='sum', margin=args.C)
    if t.cuda.is_available():
        one = one.to(args.device)

    size = len(hrtts)
    for epoch in tqdm(range(args.epochs)):

        for i in range(0, size, args.batch_size):
            j = min(size, i + args.batch_size)
            hs = [hrtt[0] for hrtt in hrtts[i: j]]
            rs = [hrtt[1] for hrtt in hrtts[i: j]]
            pos_ts = [hrtt[2] for hrtt in hrtts[i: j]]
            neg_ts = [hrtt[3] for hrtt in hrtts[i: j]]
            cor_scores, incor_scores = model(hs, rs, pos_ts, neg_ts)
            loss = -criterion(cor_scores.view(-1), incor_scores.view(-1), one)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    save_emb(model, args.dim, './data/' + args.dataset + '/' + str(args.dim) + '_emb.pkl')


def save_emb(model, dim, file):
    ent_emb = model.ent_emb.cpu().detach()
    rel_emb = model.rel_emb.cpu().detach()
    param_dict = {'ent_emb': F.normalize(ent_emb.view(-1, 1, dim), dim=[1, 2]), 'rel_emb': F.normalize(rel_emb.view(-1, 1, dim), dim=[1, 2])}
    t.save(param_dict, file)


def get_hrtts(kg_dict):
    random.seed(255)

    entities = list(kg_dict)

    hrtts = []
    for head in kg_dict:
        for r_t in kg_dict[head]:
            relation = r_t[0]
            positive_tail = r_t[1]

            while True:
                negative_tail = random.sample(entities, 1)[0]
                if [relation, negative_tail] not in kg_dict[head]:
                    hrtts.append([head, relation, positive_tail, negative_tail])
                    break

    return hrtts