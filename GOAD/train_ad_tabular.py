import numpy as np
from data_loader import Data_Loader
import opt_tc_tabular as tc
import argparse

def load_trans_data(args):
    dl = Data_Loader()
    train_real, val_real,val_fake,val_datas, test_real,test_fake,test_datas  = dl.get_dataset(args.dataset, args.c_pr)

    y_test_fscore = np.concatenate([np.zeros(len(val_real)), np.ones(len(val_fake))])
    ratio = 100.0 * len(val_real) / (len(val_real) + len(val_fake))

    y_eval_fscore = np.concatenate([np.zeros(len(test_real)), np.ones(len(test_fake))])
    eval_ratio = 100.0 * len(test_real) / (len(test_real) + len(test_fake))

    n_train, n_dims = train_real.shape
    rots = np.random.randn(args.n_rots, n_dims, args.d_out)                 # 64(?), 24(데이터의 dim 크기), 64(output하는 dim크기?)
    
    print('Calculating transforms')
    if args.dataset == 'kdd' or args.dataset == 'kdd_down' or args.dataset == 'cn7':
        # train data
        x_train = np.stack([train_real.dot(rot) for rot in rots], 2)            # (3762,24) --> (3762, 64, 64)
        # test data
        val_real_xs = np.stack([val_real.dot(rot) for rot in rots], 2)
        val_fake_xs = np.stack([val_fake.dot(rot) for rot in rots], 2)
        x_test = np.concatenate([val_real_xs, val_fake_xs])
        test_dataset = val_datas
        print('data setting......')
        return x_train, x_test,y_test_fscore,ratio, test_dataset
    else : 
        # eval data
        eval_real_xs = np.stack([test_real.dot(rot) for rot in rots], 2)
        eval_fake_xs = np.stack([test_fake.dot(rot) for rot in rots], 2)
        x_eval = np.concatenate([eval_real_xs, eval_fake_xs])
        eval_dataset = test_datas
        print('data setting......')
        return x_eval,y_eval_fscore,eval_ratio,eval_dataset


def train_anomaly_detector(args):
    x_train, x_test, y_test, ratio, test_dataset, x_eval,y_eval,eval_ratio,eval_dataset = load_trans_data(args)
    print('start....')
    tc_obj = tc.TransClassifierTabular(args)
    f_score = tc_obj.fit_trans_classifier(x_train, x_test, y_test, ratio, test_dataset)
    return f_score

def my_train_anomaly_detector(x_train, x_test, y_test, ratio, test_dataset):
    print('start....')
    tc_obj = tc.TransClassifierTabular(args)
    tc_obj.fit_trans_classifier(x_train, x_test, y_test, ratio, test_dataset)
    # f_score = tc_obj.fit_trans_classifier(x_train, x_test, y_test, ratio, test_dataset)
    # return f_score

def demo_anomaly_detector(x_eval,y_eval,eval_ratio,eval_dataset,PATH):
    print('test demo...')
    tc_obj = tc.TransClassifierTabular(args)
    f_score = tc_obj.demo_classifier(x_eval,y_eval,eval_ratio,eval_dataset,PATH)
    print(f_score)
    print('done')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--n_rots', default=64, type=int)           # 64, 256
    parser.add_argument('--batch_size', default=32, type=int)        # 32
    parser.add_argument('--n_epoch', default=25, type=int)
    parser.add_argument('--d_out', default=64, type=int)             # 4 64 32
    parser.add_argument('--dataset', default='cn7', type=str)   # kdd, kdd_down, kdd_demo, thyroid, arrhythmia, cn7, cn7_demo
    parser.add_argument('--exp', default='affine', type=str)
    parser.add_argument('--c_pr', default=0, type=int)
    parser.add_argument('--true_label', default=1, type=int)
    parser.add_argument('--ndf', default=32, type=int)
    parser.add_argument('--m', default=1, type=float)
    parser.add_argument('--lmbda', default=0.1, type=float)
    parser.add_argument('--eps', default=0, type=float)
    parser.add_argument('--n_iters', default=5, type=int)
    parser.add_argument('--demo_model', default='model_epoch_1.pth', type=str)

    args = parser.parse_args()

# python3 train_ad_tabular.py --n_rots=64 --n_epoch=25 --d_out=64 --ndf=32 --dataset=kdd 
# python3 train_ad_tabular.py --n_rots=256 --n_epoch=1 --d_out=32 --ndf=8 --dataset=thyroid

# python3 train_ad_tabular.py --n_rots=256 --batch_size=4 --n_epoch=10 --d_out=32 --ndf=8 --dataset=thyroid


    print("Dataset: ", args.dataset)
    # x_train, x_test, y_test, ratio, test_dataset, x_eval,y_eval,eval_ratio,eval_dataset = load_trans_data(args)


    if args.dataset == 'thyroid' or args.dataset == 'arrhythmia':
        n_iters = args.n_iters
        f_scores = np.zeros(n_iters)
        for i in range(n_iters):
            print('thyroid')
            # f_scores[i] = my_train_anomaly_detector(x_train, x_test, y_test, ratio, test_dataset)
        print("AVG f1_score", f_scores.mean())
    elif args.dataset == 'kdd_demo' :
        x_eval,y_eval,eval_ratio,eval_dataset = load_trans_data(args)
        PATH = 'checkpoints/kdd_down_checkpoint/' + args.demo_model
        print('demo test')
        demo_anomaly_detector(x_eval,y_eval,eval_ratio,eval_dataset,PATH)
    elif args.dataset == 'cn7_demo' :
        x_eval,y_eval,eval_ratio,eval_dataset = load_trans_data(args)
        PATH = 'checkpoints/cn7_checkpoint/' + args.demo_model
        print('demo test')
        demo_anomaly_detector(x_eval,y_eval,eval_ratio,eval_dataset,PATH)
    else:
        x_train, x_test, y_test, ratio, test_dataset = load_trans_data(args)
        # train_anomaly_detector(args)
        my_train_anomaly_detector(x_train, x_test, y_test, ratio, test_dataset)
