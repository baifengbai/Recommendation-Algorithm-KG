import numpy as np

from train import train
import argparse


if __name__ == '__main__':

    steps = 3
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
        # parser.add_argument('--H', type=int, default=1, help='H')
        # parser.add_argument('--K', type=int, default=8, help='K')
        # parser.add_argument('--l1', type=float, default=1e-2, help='L1')

        # parser.add_argument('--dataset', type=str, default='ml', help='dataset')
        # parser.add_argument('--lr', type=float, default=5e-2, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-5, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=20, help='epochs')
        # parser.add_argument("--device", type=str, default='cuda:0', help='device')
        # parser.add_argument('--dim', type=int, default=16, help='embedding size')
        # parser.add_argument('--H', type=int, default=1, help='H')
        # parser.add_argument('--K', type=int, default=8, help='K')
        # parser.add_argument('--l1', type=float, default=1e-2, help='L1')
        #
        # parser.add_argument('--dataset', type=str, default='music', help='dataset')
        # parser.add_argument('--lr', type=float, default=2e-1, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-5, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=20, help='epochs')
        # parser.add_argument("--device", type=str, default='cuda:0', help='device')
        # parser.add_argument('--dim', type=int, default=16, help='embedding size')
        # parser.add_argument('--H', type=int, default=3, help='H')
        # parser.add_argument('--K', type=int, default=8, help='K')
        # parser.add_argument('--l1', type=float, default=1e-2, help='L1')
        #
        # parser.add_argument('--dataset', type=str, default='book', help='dataset')
        # parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-5, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=20, help='epochs')
        # parser.add_argument("--device", type=str, default='cuda:0', help='device')
        # parser.add_argument('--dim', type=int, default=16, help='embedding size')
        # parser.add_argument('--H', type=int, default=1, help='H')
        # parser.add_argument('--K', type=int, default=8, help='K')
        # parser.add_argument('--l1', type=float, default=1e-2, help='L1')
        #
        # parser.add_argument('--dataset', type=str, default='movie', help='dataset')
        # parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-5, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=20, help='epochs')
        # parser.add_argument("--device", type=str, default='cuda:0', help='device')
        # parser.add_argument('--dim', type=int, default=16, help='embedding size')
        # parser.add_argument('--H', type=int, default=1, help='H')
        # parser.add_argument('--K', type=int, default=8, help='K')
        # parser.add_argument('--l1', type=float, default=1e-2, help='L1')

        parser.add_argument('--dataset', type=str, default='yelp', help='dataset')
        parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
        parser.add_argument('--l2', type=float, default=1e-5, help='L2')
        parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        parser.add_argument('--epochs', type=int, default=20, help='epochs')
        parser.add_argument("--device", type=str, default='cuda:0', help='device')
        parser.add_argument('--dim', type=int, default=16, help='embedding size')
        parser.add_argument('--H', type=int, default=1, help='H')
        parser.add_argument('--K', type=int, default=8, help='K')
        parser.add_argument('--l1', type=float, default=1e-2, help='L1')

        args = parser.parse_args()

        indicators = train(args, True)
        auc_np[step] = indicators[1]
        acc_np[step] = indicators[2]
        recall_np[step] = np.array(indicators[3])
        ndcg_np[step] = np.array(indicators[4])

    # print('AUC[\'RippleNet\'][\'job\'] =', auc_np.mean().round(3))
    # print('ACC[\'RippleNet\'][\'job\'] =', acc_np.mean().round(3))
    # print('Recall[\'RippleNet\'][\'job\'] =', recall_np.mean(axis=0).round(3).tolist())
    # print('NDCG[\'RippleNet\'][\'job\'] =', ndcg_np.mean(axis=0).round(3).tolist())

    # print('AUC[\'RippleNet\'][\'ml\'] =', auc_np.mean().round(3))
    # print('ACC[\'RippleNet\'][\'ml\'] =', acc_np.mean().round(3))
    # print('Recall[\'RippleNet\'][\'ml\'] =', recall_np.mean(axis=0).round(3).tolist())
    # print('NDCG[\'RippleNet\'][\'ml\'] =', ndcg_np.mean(axis=0).round(3).tolist())

    # print('AUC[\'RippleNet\'][\'music\'] =', auc_np.mean().round(3))
    # print('ACC[\'RippleNet\'][\'music\'] =', acc_np.mean().round(3))
    # print('Recall[\'RippleNet\'][\'music\'] =', recall_np.mean(axis=0).round(3).tolist())
    # print('NDCG[\'RippleNet\'][\'music\'] =', ndcg_np.mean(axis=0).round(3).tolist())

    # print('AUC[\'RippleNet\'][\'book\'] =', auc_np.mean().round(3))
    # print('ACC[\'RippleNet\'][\'book\'] =', acc_np.mean().round(3))
    # print('Recall[\'RippleNet\'][\'book\'] =', recall_np.mean(axis=0).round(3).tolist())
    # print('NDCG[\'RippleNet\'][\'book\'] =', ndcg_np.mean(axis=0).round(3).tolist())

    # print('AUC[\'RippleNet\'][\'movie\'] =', auc_np.mean().round(3))
    # print('ACC[\'RippleNet\'][\'movie\'] =', acc_np.mean().round(3))
    # print('Recall[\'RippleNet\'][\'movie\'] =', recall_np.mean(axis=0).round(3).tolist())
    # print('NDCG[\'RippleNet\'][\'movie\'] =', ndcg_np.mean(axis=0).round(3).tolist())

    print('AUC[\'RippleNet\'][\'yelp\'] =', auc_np.mean().round(3))
    print('ACC[\'RippleNet\'][\'yelp\'] =', acc_np.mean().round(3))
    print('Recall[\'RippleNet\'][\'yelp\'] =', recall_np.mean(axis=0).round(3).tolist())
    print('NDCG[\'RippleNet\'][\'yelp\'] =', ndcg_np.mean(axis=0).round(3).tolist())


'''





'''

