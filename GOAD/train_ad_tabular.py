import numpy as np
from data_loader import Data_Loader
import training.opt_tc_tabular as tc
import argparse
from tqdm import tqdm 
import torch 
import warnings
warnings.filterwarnings('ignore')

def load_trans_data(args):
    dl = Data_Loader()
    train_real, val_norm, val_abnorm, val_dataset, mus, sds = dl.get_dataset(args.dataset, args.c_pr)
    y_val_fscore = np.concatenate([np.zeros(len(val_norm)), np.ones(len(val_abnorm))])
    ratio = 100.0 * len(val_norm) / (len(val_norm) + len(val_abnorm))

    n_train, n_dims = train_real.shape
    print('Data shape: ',train_real.shape )
    
    rots = torch.from_numpy(np.random.randn(args.n_rots, n_dims, args.d_out)).float().cuda()       
    train_real = torch.from_numpy(train_real.astype(float)).float().cuda()  
    val_norm = torch.from_numpy(val_norm.astype(float)).float().cuda()  
    val_abnorm = torch.from_numpy(val_abnorm.astype(float)).float().cuda() 

    print('Calculating transforms')
    x_train = torch.stack([train_real.matmul(rot) for rot in tqdm(rots, desc='train data')], dim=2)
    val_norm_xs = torch.stack([val_norm.matmul(rot) for rot in tqdm(rots, desc='val normal data')], dim=2)
    val_abnorm_xs = torch.stack([val_abnorm.matmul(rot) for rot in tqdm(rots, desc='val abnormal data')], dim=2)
    x_val = torch.concatenate([val_norm_xs, val_abnorm_xs])
    print('Data set up complete')
    
    return x_train, x_val, y_val_fscore, val_dataset, ratio, mus, sds


def train_anomaly_detector(args):
    x_train, x_val, y_val, val_dataset, ratio, mus, sds = load_trans_data(args)
    print('start....')
    tc_obj = tc.TransClassifierTabular(args)
    f_score = tc_obj.fit_trans_classifier(x_train, x_val, y_val, val_dataset, ratio, mus, sds, vis=True)
    return f_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--n_rots', default=256, type=int)            # 64,  64,  256,     256
    parser.add_argument('--batch_size', default=64, type=int)        # 64
    parser.add_argument('--n_epoch', default=1, type=int)           # 25,  25,  25,      25   
    parser.add_argument('--d_out', default=32, type=int)             # 64,  64,  32,      32
    parser.add_argument('--dataset', default='thyroid', type=str)    # kdd, cn7, thyroid, arrhythmia
    parser.add_argument('--exp', default='affine', type=str)
    parser.add_argument('--c_pr', default=0, type=int)
    parser.add_argument('--true_label', default=1, type=int)
    parser.add_argument('--ndf', default=8, type=int)               # 32,  32,   8,        8
    parser.add_argument('--m', default=1, type=float)
    parser.add_argument('--lmbda', default=0.1, type=float)
    parser.add_argument('--eps', default=0, type=float)
    parser.add_argument('--n_iters', default=100, type=int)

    args = parser.parse_args()
    print("Dataset: ", args.dataset)


    if args.dataset == 'thyroid' or args.dataset == 'arrhythmia':
        n_iters = args.n_iters
        f_scores = np.zeros(n_iters)
        for i in tqdm(range(n_iters)):            # k-fold
            f_scores[i] = train_anomaly_detector(args)
        print("AVG f1_score", f_scores.mean())
    else:
        train_anomaly_detector(args)
