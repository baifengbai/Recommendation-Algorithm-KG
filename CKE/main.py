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
        parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
        parser.add_argument('--l2', type=float, default=1e-5, help='L2')
        parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        parser.add_argument('--epochs', type=int, default=50, help='epochs')
        parser.add_argument('--device', type=str, default='cuda:0', help='device')
        parser.add_argument('--dim', type=int, default=32, help='embedding size')
        parser.add_argument('--lu', type=float, default=1e-7, help='lu')
        parser.add_argument('--lv', type=float, default=1e-7, help='lv')
        parser.add_argument('--lM', type=float, default=1e-7, help='lM')
        parser.add_argument('--lr', type=float, default=1e-7, help='lr')
        parser.add_argument('--lI', type=float, default=1e-7, help='lI')

        args = parser.parse_args()
        indicators = train(args, True)
        auc_np[step] = indicators[1]
        acc_np[step] = indicators[2]
        recall_np[step] = np.array(indicators[3])
        ndcg_np[step] = np.array(indicators[4])


    # print('AUC[\'CKE\'][\'ml\'] =', auc_np.mean().round(3))
    # print('ACC[\'CKE\'][\'ml\'] =', acc_np.mean().round(3))
    # print('Recall[\'CKE\'][\'ml\'] =', recall_np.mean(axis=0).round(3).tolist())
    # print('NDCG[\'CKE\'][\'ml\'] =', ndcg_np.mean(axis=0).round(3).tolist())

    # print('AUC[\'CKE\'][\'music\'] =', auc_np.mean().round(3))
    # print('ACC[\'CKE\'][\'music\'] =', acc_np.mean().round(3))
    # print('Recall[\'CKE\'][\'music\'] =', recall_np.mean(axis=0).round(3).tolist())
    # print('NDCG[\'CKE\'][\'music\'] =', ndcg_np.mean(axis=0).round(3).tolist())

    # print('AUC[\'CKE\'][\'book\'] =', auc_np.mean().round(3))
    # print('ACC[\'CKE\'][\'book\'] =', acc_np.mean().round(3))
    # print('Recall[\'CKE\'][\'book\'] =', recall_np.mean(axis=0).round(3).tolist())
    # print('NDCG[\'CKE\'][\'book\'] =', ndcg_np.mean(axis=0).round(3).tolist())

    # print('AUC[\'CKE\'][\'movie\'] =', auc_np.mean().round(3))
    # print('ACC[\'CKE\'][\'movie\'] =', acc_np.mean().round(3))
    # print('Recall[\'CKE\'][\'movie\'] =', recall_np.mean(axis=0).round(3).tolist())
    # print('NDCG[\'CKE\'][\'movie\'] =', ndcg_np.mean(axis=0).round(3).tolist())

    print('AUC[\'CKE\'][\'yelp\'] =', auc_np.mean().round(3))
    print('ACC[\'CKE\'][\'yelp\'] =', acc_np.mean().round(3))
    print('Recall[\'CKE\'][\'yelp\'] =', recall_np.mean(axis=0).round(3).tolist())
    print('NDCG[\'CKE\'][\'yelp\'] =', ndcg_np.mean(axis=0).round(3).tolist())

'''     


'''