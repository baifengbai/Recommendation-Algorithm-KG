import numpy as np

from train import train
import argparse


if __name__ == '__main__':
    steps = 5
    auc_np = np.zeros(steps)
    acc_np = np.zeros(steps)
    recall_np = np.zeros([steps, 7])
    ndcg_np = np.zeros([steps, 7])
    for step in range(steps):
        parser = argparse.ArgumentParser()

        parser.add_argument('--dataset', type=str, default='yelp', help='dataset')
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        parser.add_argument('--l2', type=float, default=1e-5, help='L2')
        parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        parser.add_argument('--epochs', type=int, default=20, help='epochs')
        parser.add_argument('--device', type=str, default='cuda:0', help='device')
        parser.add_argument('--dim', type=int, default=32, help='embedding size')
        parser.add_argument('--L', type=int, default=1, help='L')
        parser.add_argument('--T', type=int, default=5, help='T')
        parser.add_argument('--l1', type=float, default=1e-6, help='kg loss weight')

        args = parser.parse_args()
        indicators = train(args, True)
        auc_np[step] = indicators[1]
        acc_np[step] = indicators[2]
        recall_np[step] = np.array(indicators[3])
        ndcg_np[step] = np.array(indicators[3])

    # print('AUC[\'MKR\'][\'job\'] =', auc_np.mean().round(3))
    # print('ACC[\'MKR\'][\'job\'] =', acc_np.mean().round(3))
    # print('Recall[\'MKR\'][\'job\'] =', recall_np.mean(axis=0).round(3).tolist())
    # print('NDCG[\'MKR\'][\'job\'] =', ndcg_np.mean(axis=0).round(3).tolist())

    # print('AUC[\'MKR\'][\'ml\'] =', auc_np.mean().round(3))
    # print('ACC[\'MKR\'][\'ml\'] =', acc_np.mean().round(3))
    # print('Recall[\'MKR\'][\'ml\'] =', recall_np.mean(axis=0).round(3).tolist())
    # print('NDCG[\'MKR\'][\'ml\'] =', ndcg_np.mean(axis=0).round(3).tolist())

    # print('AUC[\'MKR\'][\'music\'] =', auc_np.mean().round(3))
    # print('ACC[\'MKR\'][\'music\'] =', acc_np.mean().round(3))
    # print('Recall[\'MKR\'][\'music\'] =', recall_np.mean(axis=0).round(3).tolist())
    # print('NDCG[\'MKR\'][\'music\'] =', ndcg_np.mean(axis=0).round(3).tolist())

    # print('AUC[\'MKR\'][\'book\'] =', auc_np.mean().round(3))
    # print('ACC[\'MKR\'][\'book\'] =', acc_np.mean().round(3))
    # print('Recall[\'MKR\'][\'book\'] =', recall_np.mean(axis=0).round(3).tolist())
    # print('NDCG[\'MKR\'][\'book\'] =', ndcg_np.mean(axis=0).round(3).tolist())

    print('AUC[\'MKR\'][\'movie\'] =', auc_np.mean().round(3))
    print('ACC[\'MKR\'][\'movie\'] =', acc_np.mean().round(3))
    print('Recall[\'MKR\'][\'movie\'] =', recall_np.mean(axis=0).round(3).tolist())
    print('NDCG[\'MKR\'][\'movie\'] =', ndcg_np.mean(axis=0).round(3).tolist())

    # print('AUC[\'MKR\'][\'yelp\'] =', auc_np.mean().round(3))
    # print('ACC[\'MKR\'][\'yelp\'] =', acc_np.mean().round(3))
    # print('Recall[\'MKR\'][\'yelp\'] =', recall_np.mean(axis=0).round(3).tolist())
    # print('NDCG[\'MKR\'][\'yelp\'] =', ndcg_np.mean(axis=0).round(3).tolist())

'''

'''