import numpy as np
import pandas as pd
import random


def get_uvvs(pairs):
    positive_dict = {}
    negative_dict = {}
    for pair in pairs:
        user = pair[0]
        item = pair[1]
        label = pair[2]
        if label == 1:
            if user not in positive_dict:
                positive_dict[user] = []

            positive_dict[user].append(item)
        else:
            if user not in negative_dict:
                negative_dict[user] = []

            negative_dict[user].append(item)
    data = []
    for user in positive_dict:
        size = len(positive_dict[user])
        # print(len(positive_dict[user]), len(negative_dict[user]))
        for i in range(size):
            pos_item = positive_dict[user][i]
            neg_item = negative_dict[user][i]
            data.append([user, pos_item, neg_item])

    return data


def get_hrtts(kg_dict):
    print('get hrtts...')
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


def load_kg(file):
    # print(rel_dict)
    print('load_kg...')
    edges = pd.read_csv(file + 'kg.txt', delimiter='\t', header=None).values
    kg_dict = {}
    relation_set = set()
    entity_set = set()
    for edge in edges:
        head = edge[0]
        tail = edge[1]
        relation = edge[2]

        if head not in kg_dict:
            kg_dict[head] = []
        kg_dict[head].append([relation, tail])

        entity_set.add(head)
        entity_set.add(tail)
        relation_set.add(relation)

    return kg_dict, len(entity_set), len(relation_set)


def load_ratings(data_dir):

    data_np = pd.read_csv(data_dir + 'ratings.txt', delimiter='\t', header=None).values

    return data_np


def convert_dict(ratings_np):

    positive_records = dict()
    negative_records = dict()

    for pair in ratings_np:
        user = pair[0]
        item = pair[1]
        label = pair[2]

        if label == 1:
            if user not in positive_records:
                positive_records[user] = []
            positive_records[user].append(item)
        else:
            if user not in negative_records:
                negative_records[user] = []
            negative_records[user].append(item)

    return positive_records, negative_records


def load_data(args):

    data_dir = '../data/' + args.dataset + '/'
    ratings_np = load_ratings(data_dir)
    n_user = len(set(ratings_np[:, 0]))
    kg_dict, n_entity, n_relation = load_kg(data_dir)
    data = [n_entity + n_user, n_relation, kg_dict]

    return data