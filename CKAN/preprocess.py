from data_loader import load_kg, load_ratings, data_split, get_records
from tqdm import tqdm
import numpy as np


def get_ripple_set(train_dict, kg_dict, H, size):

    np.random.seed(255)

    ripple_set_dict = {user: [] for user in train_dict}

    for u in tqdm(train_dict):

        next_e_list = train_dict[u]
        replace = len(train_dict[u]) < size
        indices = np.random.choice(len(train_dict[u]), size, replace=replace)
        ripple_set_dict[u].append([train_dict[u][i] for i in indices])
        for h in range(H):
            h_head_list = []
            h_relation_list = []
            h_tail_list = []
            for head in next_e_list:
                if head not in kg_dict:
                    continue
                for rt in kg_dict[head]:
                    relation = rt[0]
                    tail = rt[1]
                    h_head_list.append(head)
                    h_relation_list.append(relation)
                    h_tail_list.append(tail)

            if len(h_head_list) == 0:
                h_head_list = ripple_set_dict[u][-1][0]
                h_relation_list = ripple_set_dict[u][-1][1]
                h_tail_list = ripple_set_dict[u][-1][0]
            else:
                replace = len(h_head_list) < size
                indices = np.random.choice(len(h_head_list), size, replace=replace)
                h_head_list = [h_head_list[i] for i in indices]
                h_relation_list = [h_relation_list[i] for i in indices]
                h_tail_list = [h_tail_list[i] for i in indices]

            ripple_set_dict[u].append((h_head_list, h_relation_list, h_tail_list))

            next_e_list = ripple_set_dict[u][-1][2]

    return ripple_set_dict


def get_item_record(item_set, train_records):

    ripple_sets = dict()

    for item in item_set:
        item_ripple_set = {item}
        for items in train_records.values():
            if item in items:
                item_ripple_set.update(items)
        ripple_sets[item] = list(item_ripple_set)

    return ripple_sets


def process(data_set_name):

    data_dir = '../data/' + data_set_name + '/'
    ratings = load_ratings(data_dir)
    item_set = set([i[1] for i in ratings])
    train_set, eval_set, test_set = data_split(ratings)
    user_history_dict = get_records(train_set)
    all_records = user_history_dict.copy()
    all_records.update(get_item_record(item_set, user_history_dict))
    kg_dict, _, _ = load_kg(data_dir)
    for i in range(1, 7):
        ripple_set_dict = get_ripple_set(all_records, kg_dict, 4, 2**i)
        np.save(file='./data/'+data_set_name+'/'+str(2**i) + '_ripple_set_dict.npy', arr=ripple_set_dict)


if __name__ == '__main__':

    # process('job')
    # process('music')
    # process('book')
    # process('ml')
    process('movie')
    process('yelp')
