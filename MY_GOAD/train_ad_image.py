import argparse
import utils.transformations as ts
import training.opt_tc as tc
import numpy as np
from data_loader import Data_Loader
import torch 
import os 
from collections import defaultdict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def transform_data(data, trans):
    trans_inds = np.tile(np.arange(trans.n_transforms), len(data))
    trans_data = trans.transform_batch(np.repeat(np.array(data), trans.n_transforms, axis=0), trans_inds)
    return trans_data, trans_inds

# 데이터 로드
def load_trans_data(args, trans):
    dl = Data_Loader()                                                                # args.class_ind: 타겟하는 클래스, 양품 선정
    x_train, x_valid, y_valid, x_test, y_test, class_name = dl.get_dataset(args.dataset, true_label=args.class_ind)
    x_train_trans, labels = transform_data(x_train, trans)  # (5000, 32, 32, 3) -> (360000, 32, 32, 3)
    x_valid, _ = transform_data(x_valid, trans)
    x_test, _ = transform_data(x_test, trans)
    
    x_train_trans = x_train_trans.transpose(0, 3, 1, 2)
    
    x_valid_trans = x_valid.transpose(0, 3, 1, 2)
    y_valid = (np.array(y_valid) == args.class_ind)     # target normal 번호는 True 나머지 abnormal들은 False로 바꿔줌
    
    x_test_trans = x_test.transpose(0, 3, 1, 2)
    y_test = (np.array(y_test) == args.class_ind)     # target normal 번호는 True 나머지 abnormal들은 False로 바꿔줌
    
    print(len(x_train_trans), len(x_valid_trans), len(x_test_trans))
    return class_name, x_train_trans, x_valid_trans, y_valid, x_test_trans, y_test

# 훈련
def train_anomaly_detector(args):
    transformer = ts.get_transformer(args.type_trans)
    class_name, x_train, x_valid, y_valid, x_test, y_test = load_trans_data(args, transformer)
    tc_obj = tc.TransClassifier(transformer.n_transforms, device, args)
    vaild_best_auc, normal_means, optimal_threshold = tc_obj.fit_trans_classifier(x_train, x_valid, y_valid, class_name)
    test_performance, f1 = tc_obj.test(x_test, y_test, normal_means, optimal_threshold)
    return vaild_best_auc, test_performance



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wide Residual Networks')
    # Model options
    parser.add_argument('--depth', default=10, type=int)
    parser.add_argument('--widen-factor', default=4, type=int)

    # Training options
    parser.add_argument('--batch_size', default=288*3, type=int)  # 288     # 8 또는 72의 배수 
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float)
    parser.add_argument('--epochs', default=50, type=int)    # 16
    parser.add_argument('--all_train', default=False, type=bool)    # 10개 클래스 모두 훈련시키고 싶으면 True
    parser.add_argument('--class_ind', default=7, type=int)

    # Trans options
    parser.add_argument('--type_trans', default='simple', type=str)    # complicated or simple

    # CT options
    parser.add_argument('--lmbda', default=0.1, type=float)
    parser.add_argument('--m', default=0.1, type=float)               # tc_loss 계산때 사용, cifar 이미지 사용때는 0.1, tabular 데이터는 1
    parser.add_argument('--reg', default=True, type=bool)
    parser.add_argument('--eps', default=0, type=float)

    # Exp options
    parser.add_argument('--dataset', default='cifar10', type=str)
    
    # save model 
    parser.add_argument('--save_path', default='./checkpoints/cifa10', type=str)
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    
    
    print("Dataset: CIFAR10")
    class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 10개 클래스에 대한 각각의 모델을 만들고 싶으면 
    if args.all_train:
        results = defaultdict()
        for i in range(len(class_name)):
            args.class_ind = i
            print("Target Normal class:", args.class_ind, class_name[i])
            valid_best_auc, test_performance = train_anomaly_detector(args)
            valid_best_auc = '{:.2f}'.format(100*valid_best_auc)
            test_performance = '{:.2f}'.format(100*test_performance)
            results[class_name[i]] = [valid_best_auc, test_performance]
            print(results)

        print(results)
    else:
        print("Target Normal class:", args.class_ind, class_name[args.class_ind])
        valid_best_auc, test_performance = train_anomaly_detector(args)
        valid_best_auc = '{:.2f}'.format(100*valid_best_auc)
        test_performance = '{:.2f}'.format(100*test_performance)
        print(valid_best_auc)
        print(test_performance)
    
'''
CiFAR10

AUC - vaild, test
airplane : 71.90%, 72.49%
automobile : 94.00%, 91.92%
bird : 71.69%, 69.83% 
cat : 61.73%, 60.88%
deer : 75.12%, 74.07%
dog : 79.38%, 78.80%
frog : 69.22%, 68.93%
horse : 93.08%, 93.32%
ship : 92.38%, 92.65%
truck : 89.35%, 87.67%


'''