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

        # parser.add_argument('--dataset', type=str, default='job', help='dataset')
        # parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-5, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=20, help='epochs')
        # parser.add_argument('--device', type=str, default='cuda:2', help='device')
        # parser.add_argument('--dim', type=int, default=16, help='embedding size')
        # parser.add_argument('--L', type=int, default=1, help='H')
        # parser.add_argument('--K', type=int, default=8, help='K')
        # parser.add_argument('--agg', type=str, default='concat', help='K')

        # parser.add_argument('--dataset', type=str, default='ml', help='dataset')
        # parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-5, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=50, help='epochs')
        # parser.add_argument('--device', type=str, default='cuda:2', help='device')
        # parser.add_argument('--dim', type=int, default=16, help='embedding size')
        # parser.add_argument('--L', type=int, default=1, help='H')
        # parser.add_argument('--K', type=int, default=8, help='K')
        # parser.add_argument('--agg', type=str, default='concat', help='K')
        #
        # parser.add_argument('--dataset', type=str, default='music', help='dataset')
        # parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-5, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=20, help='epochs')
        # parser.add_argument('--device', type=str, default='cuda:2', help='device')
        # parser.add_argument('--dim', type=int, default=16, help='embedding size')
        # parser.add_argument('--L', type=int, default=3, help='H')
        # parser.add_argument('--K', type=int, default=8, help='K')
        # parser.add_argument('--agg', type=str, default='concat', help='K')
        #
        # parser.add_argument('--dataset', type=str, default='book', help='dataset')
        # parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-5, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=50, help='epochs')
        # parser.add_argument('--device', type=str, default='cuda:2', help='device')
        # parser.add_argument('--dim', type=int, default=16, help='embedding size')
        # parser.add_argument('--L', type=int, default=1, help='H')
        # parser.add_argument('--K', type=int, default=8, help='K')
        # parser.add_argument('--agg', type=str, default='concat', help='K')

        # parser.add_argument('--dataset', type=str, default='movie', help='dataset')
        # parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-5, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=20, help='epochs')
        # parser.add_argument('--device', type=str, default='cuda:0', help='device')
        # parser.add_argument('--dim', type=int, default=16, help='embedding size')
        # parser.add_argument('--L', type=int, default=3, help='H')
        # parser.add_argument('--K', type=int, default=8, help='K')
        # parser.add_argument('--agg', type=str, default='concat', help='K')

        parser.add_argument('--dataset', type=str, default='yelp', help='dataset')
        parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
        parser.add_argument('--l2', type=float, default=1e-5, help='L2')
        parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        parser.add_argument('--epochs', type=int, default=50, help='epochs')
        parser.add_argument('--device', type=str, default='cuda:2', help='device')
        parser.add_argument('--dim', type=int, default=16, help='embedding size')
        parser.add_argument('--L', type=int, default=2, help='H')
        parser.add_argument('--K', type=int, default=8, help='K')
        parser.add_argument('--agg', type=str, default='concat', help='K')

        args = parser.parse_args()

        indicators = train(args, True)
        auc_np[step] = indicators[1]
        acc_np[step] = indicators[2]
        recall_np[step] = np.array(indicators[3])
        ndcg_np[step] = np.array(indicators[4])

    # print('AUC[\'CKAN\'][\'job\'] =', auc_np.mean().round(3))
    # print('ACC[\'CKAN\'][\'job\'] =', acc_np.mean().round(3))
    # print('Recall[\'CKAN\'][\'job\'] =', recall_np.mean(axis=0).round(3).tolist())
    # print('NDCG[\'CKAN\'][\'job\'] =', ndcg_np.mean(axis=0).round(3).tolist())

    # print('AUC[\'CKAN\'][\'ml\'] =', auc_np.mean().round(3))
    # print('ACC[\'CKAN\'][\'ml\'] =', acc_np.mean().round(3))
    # print('Recall[\'CKAN\'][\'ml\'] =', recall_np.mean(axis=0).round(3).tolist())
    # print('NDCG[\'CKAN\'][\'ml\'] =', ndcg_np.mean(axis=0).round(3).tolist())

    # print('AUC[\'CKAN\'][\'music\'] =', auc_np.mean().round(3))
    # print('ACC[\'CKAN\'][\'music\'] =', acc_np.mean().round(3))
    # print('Recall[\'CKAN\'][\'music\'] =', recall_np.mean(axis=0).round(3).tolist())
    # print('NDCG[\'CKAN\'][\'music\'] =', ndcg_np.mean(axis=0).round(3).tolist())

    # print('AUC[\'CKAN\'][\'book\'] =', auc_np.mean().round(3))
    # print('ACC[\'CKAN\'][\'book\'] =', acc_np.mean().round(3))
    # print('Recall[\'CKAN\'][\'book\'] =', recall_np.mean(axis=0).round(3).tolist())
    # print('NDCG[\'CKAN\'][\'book\'] =', ndcg_np.mean(axis=0).round(3).tolist())

    # print('AUC[\'CKAN\'][\'movie\'] =', auc_np.mean().round(3))
    # print('ACC[\'CKAN\'][\'movie\'] =', acc_np.mean().round(3))
    # print('Recall[\'CKAN\'][\'movie\'] =', recall_np.mean(axis=0).round(3).tolist())
    # print('NDCG[\'CKAN\'][\'movie\'] =', ndcg_np.mean(axis=0).round(3).tolist())

    print('AUC[\'CKAN\'][\'yelp\'] =', auc_np.mean().round(3))
    print('ACC[\'CKAN\'][\'yelp\'] =', acc_np.mean().round(3))
    print('Recall[\'CKAN\'][\'yelp\'] =', recall_np.mean(axis=0).round(3).tolist())
    print('NDCG[\'CKAN\'][\'yelp\'] =', ndcg_np.mean(axis=0).round(3).tolist())

'''




'''