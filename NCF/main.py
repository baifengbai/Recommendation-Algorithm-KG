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
                parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
                parser.add_argument('--l2', type=float, default=1e-5, help='L2')
                parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
                parser.add_argument('--epochs', type=int, default=100, help='epochs')
                parser.add_argument('--device', type=str, default='cuda:0', help='device')
                parser.add_argument('--dim', type=int, default=32, help='embedding size')
                parser.add_argument('--is_pre', type=bool, default=True, help='pretrain')

                args = parser.parse_args()
                indicators = train(args, True)
                auc_np[step] = indicators[1]
                acc_np[step] = indicators[2]
                recall_np[step] = np.array(indicators[3])
                ndcg_np[step] = np.array(indicators[4])

        # print('AUC[\'NCF\'][\'job\'] =', auc_np.mean().round(3))
        # print('ACC[\'NCF\'][\'job\'] =', acc_np.mean().round(3))
        # print('Recall[\'NCF\'][\'job\'] =', recall_np.mean(axis=0).round(3).tolist())
        # print('NDCG[\'NCF\'][\'job\'] =', ndcg_np.mean(axis=0).round(3).tolist())

        # print('AUC[\'NCF\'][\'ml\'] =', auc_np.mean().round(3))
        # print('ACC[\'NCF\'][\'ml\'] =', acc_np.mean().round(3))
        # print('Recall[\'NCF\'][\'ml\'] =', recall_np.mean(axis=0).round(3).tolist())
        # print('NDCG[\'NCF\'][\'ml\'] =', ndcg_np.mean(axis=0).round(3).tolist())

        # print('AUC[\'NCF\'][\'music\'] =', auc_np.mean().round(3))
        # print('ACC[\'NCF\'][\'music\'] =', acc_np.mean().round(3))
        # print('Recall[\'NCF\'][\'music\'] =', recall_np.mean(axis=0).round(3).tolist())
        # print('NDCG[\'NCF\'][\'music\'] =', ndcg_np.mean(axis=0).round(3).tolist())

        # print('AUC[\'NCF\'][\'book\'] =', auc_np.mean().round(3))
        # print('ACC[\'NCF\'][\'book\'] =', acc_np.mean().round(3))
        # print('Recall[\'NCF\'][\'book\'] =', recall_np.mean(axis=0).round(3).tolist())
        # print('NDCG[\'NCF\'][\'book\'] =', ndcg_np.mean(axis=0).round(3).tolist())

        # print('AUC[\'NCF\'][\'movie\'] =', auc_np.mean().round(3))
        # print('ACC[\'NCF\'][\'movie\'] =', acc_np.mean().round(3))
        # print('Recall[\'NCF\'][\'movie\'] =', recall_np.mean(axis=0).round(3).tolist())
        # print('NDCG[\'NCF\'][\'movie\'] =', ndcg_np.mean(axis=0).round(3).tolist())

        print('AUC[\'NCF\'][\'yelp\'] =', auc_np.mean().round(3))
        print('ACC[\'NCF\'][\'yelp\'] =', acc_np.mean().round(3))
        print('Recall[\'NCF\'][\'yelp\'] =', recall_np.mean(axis=0).round(3).tolist())
        print('NDCG[\'NCF\'][\'yelp\'] =', ndcg_np.mean(axis=0).round(3).tolist())

'''


'''