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
                # parser.add_argument("--device", type=str, default='cuda:0', help='device')
                # parser.add_argument('--dim', type=int, default=16, help='embedding size')
                # parser.add_argument('--n_neighbors', type=int, default=8, help='the number of neighbors')
                # parser.add_argument('--n_iter', type=int, default=2, help='the number of layers of KGCN')
                #
                # parser.add_argument('--dataset', type=str, default='ml', help='dataset')
                # parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
                # parser.add_argument('--l2', type=float, default=1e-5, help='L2')
                # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
                # parser.add_argument('--epochs', type=int, default=20, help='epochs')
                # parser.add_argument("--device", type=str, default='cuda:0', help='device')
                # parser.add_argument('--dim', type=int, default=16, help='embedding size')
                # parser.add_argument('--n_neighbors', type=int, default=8, help='the number of neighbors')
                # parser.add_argument('--n_iter', type=int, default=1, help='the number of layers of KGCN')
                #
                # parser.add_argument('--dataset', type=str, default='music', help='dataset')
                # parser.add_argument('--lr', type=float, default=5e-2, help='learning rate')
                # parser.add_argument('--l2', type=float, default=1e-5, help='L2')
                # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
                # parser.add_argument('--epochs', type=int, default=50, help='epochs')
                # parser.add_argument("--device", type=str, default='cuda:0', help='device')
                # parser.add_argument('--dim', type=int, default=16, help='embedding size')
                # parser.add_argument('--n_neighbors', type=int, default=8, help='the number of neighbors')
                # parser.add_argument('--n_iter', type=int, default=1, help='the number of layers of KGCN')

                # parser.add_argument('--dataset', type=str, default='book', help='dataset')
                # parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
                # parser.add_argument('--l2', type=float, default=1e-5, help='L2')
                # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
                # parser.add_argument('--epochs', type=int, default=20, help='epochs')
                # parser.add_argument("--device", type=str, default='cuda:0', help='device')
                # parser.add_argument('--dim', type=int, default=16, help='embedding size')
                # parser.add_argument('--n_neighbors', type=int, default=8, help='the number of neighbors')
                # parser.add_argument('--n_iter', type=int, default=2, help='the number of layers of KGCN')

                # parser.add_argument('--dataset', type=str, default='movie', help='dataset')
                # parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
                # parser.add_argument('--l2', type=float, default=1e-5, help='L2')
                # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
                # parser.add_argument('--epochs', type=int, default=20, help='epochs')
                # parser.add_argument("--device", type=str, default='cuda:0', help='device')
                # parser.add_argument('--dim', type=int, default=16, help='embedding size')
                # parser.add_argument('--n_neighbors', type=int, default=8, help='the number of neighbors')
                # parser.add_argument('--n_iter', type=int, default=1, help='the number of layers of KGCN')

                parser.add_argument('--dataset', type=str, default='yelp', help='dataset')
                parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
                parser.add_argument('--l2', type=float, default=1e-5, help='L2')
                parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
                parser.add_argument('--epochs', type=int, default=10, help='epochs')
                parser.add_argument("--device", type=str, default='cuda:0', help='device')
                parser.add_argument('--dim', type=int, default=16, help='embedding size')
                parser.add_argument('--n_neighbors', type=int, default=8, help='the number of neighbors')
                parser.add_argument('--n_iter', type=int, default=1, help='the number of layers of KGCN')

                args = parser.parse_args()

                indicators = train(args, True)
                auc_np[step] = indicators[1]
                acc_np[step] = indicators[2]
                recall_np[step] = np.array(indicators[3])
                ndcg_np[step] = np.array(indicators[4])
        #
        # print('AUC[\'KGCN\'][\'job\'] =', auc_np.mean().round(3))
        # print('ACC[\'KGCN\'][\'job\'] =', acc_np.mean().round(3))
        # print('Recall[\'KGCN\'][\'job\'] =', recall_np.mean(axis=0).round(3).tolist())
        # print('NDCG[\'KGCN\'][\'job\'] =', ndcg_np.mean(axis=0).round(3).tolist())

        # print('AUC[\'KGCN\'][\'ml\'] =', auc_np.mean().round(3))
        # print('ACC[\'KGCN\'][\'ml\'] =', acc_np.mean().round(3))
        # print('Recall[\'KGCN\'][\'ml\'] =', recall_np.mean(axis=0).round(3).tolist())
        # print('NDCG[\'KGCN\'][\'ml\'] =', ndcg_np.mean(axis=0).round(3).tolist())

        # print('AUC[\'KGCN\'][\'music\'] =', auc_np.mean().round(3))
        # print('ACC[\'KGCN\'][\'music\'] =', acc_np.mean().round(3))
        # print('Recall[\'KGCN\'][\'music\'] =', recall_np.mean(axis=0).round(3).tolist())
        # print('NDCG[\'KGCN\'][\'music\'] =', ndcg_np.mean(axis=0).round(3).tolist())

        # print('AUC[\'KGCN\'][\'book\'] =', auc_np.mean().round(3))
        # print('ACC[\'KGCN\'][\'book\'] =', acc_np.mean().round(3))
        # print('Recall[\'KGCN\'][\'book\'] =', recall_np.mean(axis=0).round(3).tolist())
        # print('NDCG[\'KGCN\'][\'book\'] =', ndcg_np.mean(axis=0).round(3).tolist())

        # print('AUC[\'KGCN\'][\'movie\'] =', auc_np.mean().round(3))
        # print('ACC[\'KGCN\'][\'movie\'] =', acc_np.mean().round(3))
        # print('Recall[\'KGCN\'][\'movie\'] =', recall_np.mean(axis=0).round(3).tolist())
        # print('NDCG[\'KGCN\'][\'movie\'] =', ndcg_np.mean(axis=0).round(3).tolist())

        print('AUC[\'KGCN\'][\'yelp\'] =', auc_np.mean().round(3))
        print('ACC[\'KGCN\'][\'yelp\'] =', acc_np.mean().round(3))
        print('Recall[\'KGCN\'][\'yelp\'] =', recall_np.mean(axis=0).round(3).tolist())
        print('NDCG[\'KGCN\'][\'yelp\'] =', ndcg_np.mean(axis=0).round(3).tolist())



'''
'''


