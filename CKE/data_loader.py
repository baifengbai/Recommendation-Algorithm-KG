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


def load_kg(file, train_set):
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

    n_relation = len(relation_set)
    n_entity = len(entity_set)
    for pair in train_set:
        user = pair[0]
        item = pair[1]
        label = 1
        if label == 1:
            head = user
            tail = item
            relation = n_relation

            if head not in kg_dict:
                kg_dict[head] = []

            if tail not in kg_dict:
                kg_dict[tail] = []

            kg_dict[head].append([relation, tail])
            kg_dict[tail].append([relation+1, head])
            entity_set.add(head)
            entity_set.add(tail)
            relation_set.add(relation)
            relation_set.add(relation + 1)

    return kg_dict, n_entity, len(relation_set)


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


def data_split(ratings_np):
    # print('data split...')
    np.random.seed(255)
    random.seed(255)
    positive_records, negative_records = convert_dict(ratings_np)
    train_set = []
    eval_set = []
    test_set = []
    for user in positive_records:
        pos_record = positive_records[user]
        neg_record = negative_records[user]
        size = len(pos_record)

        eval_indices = np.random.choice(size, int(size * 0.2), replace=False)
        rem_indices = list(set(range(size)) - set(eval_indices))

        test_indices = np.random.choice(rem_indices, int(size * 0.2), replace=False)
        train_indices = list(set(rem_indices) - set(test_indices))

        train_set.extend([user, pos_record[i], 1] for i in train_indices)
        train_set.extend([user, neg_record[i], 0] for i in train_indices)

        eval_set.extend([user, pos_record[i], 1] for i in eval_indices)
        eval_set.extend([user, neg_record[i], 0] for i in eval_indices)

        test_set.extend([user, pos_record[i], 1] for i in test_indices)
        test_set.extend([user, neg_record[i], 0] for i in test_indices)

    random.shuffle(train_set)
    random.shuffle(eval_set)
    random.shuffle(test_set)

    return train_set, eval_set, test_set


def get_rec(train_records, eval_records, test_records, item_set):
    np.random.seed(255)
    rec = dict()
    user_set = set(test_records.keys())
    users = np.random.choice(list(user_set), 50, replace=False)
    for user in users:
        rec[user] = list(set(item_set))
        # print(len(rec[user]))
    return rec


def get_records(data_set):

    records = dict()

    for pair in data_set:
        user = pair[0]
        item = pair[1]
        label = pair[2]

        if label == 1:
            if user not in records:
                records[user] = []

            records[user].append(item)

    return records


def load_data(dataset):
    data_dir = '../data/' + dataset + '/'
    ratings_np = load_ratings(data_dir)
    train_set, eval_set, test_set = data_split(ratings_np)

    train_records = get_records(train_set)
    eval_records = get_records(eval_set)
    test_records = get_records(test_set)

    user_set = set(ratings_np[:, 0])
    item_set = set(ratings_np[:, 1])

    rec = get_rec(train_records, eval_records, test_records, item_set)
    kg_dict, n_entity, n_relation = load_kg(data_dir, train_set)
    n_entity = n_entity + len(user_set)
    data = [n_entity, n_relation, train_set, eval_set, test_set, rec, kg_dict]

    return data
