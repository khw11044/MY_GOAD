import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import fcnet as model
from sklearn.metrics import precision_recall_fscore_support as prf
import pandas as pd
from graph import make_graph, draw_pr_curve, draw_roc_curve
import os
import matplotlib.pyplot as plt

def tc_loss(zs, m):                    # zs가 x_i           m = 1 : clusters간 거리를 정규화하는 margin
    means = zs.mean(0).unsqueeze(0)    # means : c_m 중심점
    res = ((zs.unsqueeze(2) - means.unsqueeze(1)) ** 2).sum(-1)                 # ||f(T(x_i,m)) - c_m||^2
    pos = torch.diagonal(res, dim1=1, dim2=2)                                   # max(f(T(x_i,m)) - c_m)
    offset = torch.diagflat(torch.ones(zs.size(1))).unsqueeze(0).cuda() * 1e6           # M * episilom
    neg = (res + offset).min(-1)[0]                                             # min(||f(T(x_i,m)) - c_m'||^2)
    loss = torch.clamp(pos + m - neg, min=0).mean()                             # Sigma max(||f(T(x_i,m)) - c_m||^2 + s - min(||f(T(x_i,m)) - c_m||^2), min=0)
    return loss

def f_score(scores, labels, ratio):                         # 이게 아웃풋이자 평가 확인 
    thresh = np.percentile(scores, ratio)
    y_pred = (scores >= thresh).astype(int)                 # threshold보다 높으면 
    y_true = labels.astype(int)
    precision, recall, f_score, support = prf(y_true, y_pred, average='binary')

    print('thresh : ',thresh)
    # print('precision, recall, f_score, support', round(precision,3), round(recall,3), round(f_score,3), support)

    return f_score, precision, thresh, y_pred, y_true

def save_model(ds,epoch,model,optimizer,er,thresh):
    first_folder = "checkpoints"
    folder_name = first_folder + "/" + ds + "_checkpoint/"
    model_out_path = folder_name + "model_epoch_{}.pth".format(epoch)
    if not os.path.exists(first_folder):
        os.mkdir(first_folder)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'er': er,
                'thresh': thresh
                },model_out_path)


def save_scv(folder_name, ds, test_data, val_probs_rots, thresh, y_pred, epoch):
    if ds == 'kdd_demo' or ds == 'cn7_demo':
        file_out_path = folder_name + "demo_data_result{}.csv".format(epoch)
    else:
        file_out_path = folder_name + "test_data_result{}.csv".format(epoch)
    y_pred = y_pred.reshape(-1,1)
    scores = val_probs_rots.reshape(-1,1)
    threshs = np.full(y_pred.shape,thresh)
    test_data_result = np.hstack((test_data,y_pred,scores,threshs))
    df_y_test = pd.DataFrame(test_data_result)
    # df_y_test.to_csv(file_out_path, index=None, header=None)
    df_y_test.to_csv(file_out_path, index=None)



class TransClassifierTabular():
    def __init__(self, args):
        self.ds = args.dataset
        self.m = args.m
        self.lmbda = args.lmbda
        self.batch_size = args.batch_size
        self.ndf = args.ndf
        self.n_rots = args.n_rots
        self.d_out = args.d_out
        self.eps = args.eps

        self.n_epoch = args.n_epoch
        # if args.dataset == "thyroid" or args.dataset == "arrhythmia":
        #     self.netC = model.netC1(self.d_out, self.ndf, self.n_rots).cuda()
        # else:
        #     self.netC = model.netC5(self.d_out, self.ndf, self.n_rots).cuda()
        self.netC = model.netC5(self.d_out, self.ndf, self.n_rots).cuda()           # 64 32 64
        model.weights_init(self.netC)
        self.optimizerC = optim.Adam(self.netC.parameters(), lr=args.lr, betas=(0.5, 0.999))

    def demo_classifier(self,x_eval,y_eval,eval_ratio,eval_dataset,PATH):
        dir = "checkpoints" +"/"+ self.ds + "demo_result/"
        if not os.path.exists(dir):
            os.mkdir(dir)
     
        model = self.netC
        optimizer = self.optimizerC
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['er']
        thresh = checkpoint['thresh']

        model.eval()
        sum_zs = torch.zeros((self.ndf, self.n_rots)).cuda()

        with torch.no_grad():
            val_probs_rots = np.zeros((len(x_eval), self.n_rots))
            # for i in range(0, len(x_eval), 4):
            #     batch_range = min(4, len(x_eval) - i)
            #     idx = np.arange(batch_range) + i
            xs = torch.from_numpy(x_eval).float().cuda()
            zs, fs = model(xs)
            sum_zs = sum_zs + zs.mean(0)
            means = sum_zs.t() 
            means = means.unsqueeze(0) #/len(x_eval)
            zs = zs.permute(0, 2, 1) 
            diffs = ((zs.unsqueeze(2) - means) ** 2).sum(-1)
            diffs_eps = self.eps * torch.ones_like(diffs)
            diffs = torch.max(diffs, diffs_eps)
            logp_sz = torch.nn.functional.log_softmax(-diffs, dim=2)
            val_probs_rots = -torch.diagonal(logp_sz, 0, 1, 2).cpu().data.numpy()

                
            val_probs_rots = val_probs_rots.sum(1)
            f1_score, precision, new_thresh, y_pred, y_true = f_score(val_probs_rots, y_eval, eval_ratio) 
            print(f1_score, precision, new_thresh)
            print(y_pred, y_true)
            save_scv(dir,self.ds,eval_dataset,np.round(val_probs_rots,3),np.round(new_thresh,3),y_pred, epoch)
            make_graph(dir,self.ds,epoch,thresh) # 새로운 make_graph함수 말들면 됨 demo 데이터가 100개니까 다 확인하는걸로 

        return f1_score

    def fit_trans_classifier(self, train_xs, x_test, y_test, ratio, test_data):
        labels = torch.arange(self.n_rots).unsqueeze(0).expand((self.batch_size, self.n_rots)).long().cuda()
        celoss = nn.CrossEntropyLoss()
        print('Training')
        error_list = []
        model = self.netC
        optimizer = self.optimizerC
        
        first_folder = "checkpoints"
        dir = first_folder + "/" + self.ds + "_checkpoint/"
        
        if not os.path.exists(first_folder):
            os.mkdir(first_folder)
        if not os.path.exists(dir):
            os.mkdir(dir)
        
        for epoch in range(self.n_epoch):
            model.train()
            rp = np.random.permutation(len(train_xs))
            n_batch = 0
            sum_zs = torch.zeros((self.ndf, self.n_rots)).cuda()

            for i in range(0, len(train_xs), self.batch_size):
                model.zero_grad()
                batch_range = min(self.batch_size, len(train_xs) - i)
                train_labels = labels
                if batch_range == len(train_xs) - i:
                    train_labels = torch.arange(self.n_rots).unsqueeze(0).expand((len(train_xs) - i, self.n_rots)).long().cuda()
                idx = np.arange(batch_range) + i
                xs = torch.from_numpy(train_xs[rp[idx]]).float().cuda()
                tc_zs, ce_zs = model(xs)
                sum_zs = sum_zs + tc_zs.mean(0)
                tc_zs = tc_zs.permute(0, 2, 1)  # 차원 변경

                loss_ce = celoss(ce_zs, train_labels)
                er = self.lmbda * tc_loss(tc_zs, self.m) + loss_ce
                er.backward()
                optimizer.step()
                n_batch += 1

                if i % 100 == 0:     # self.batch_size
                    print('===> Epoch [{}] ({} / {}) : loss : {:.5}'.format(epoch,i, len(train_xs), er.item()))     # loss도 나중에 저장합시다.
                    error_list.append(er.item())

            means = sum_zs.t() / n_batch
            means = means.unsqueeze(0)

            

            # ----------------------------------------------평가 ----------------------------------------------
            print('==========Evaluation==========')
            model.eval()

            with torch.no_grad():
                val_probs_rots = np.zeros((len(y_test), self.n_rots))
                for i in range(0, len(x_test), self.batch_size):
                    batch_range = min(self.batch_size, len(x_test) - i)
                    idx = np.arange(batch_range) + i
                    xs = torch.from_numpy(x_test[idx]).float().cuda()
                    zs, fs = model(xs)
                    zs = zs.permute(0, 2, 1)
                    diffs = ((zs.unsqueeze(2) - means) ** 2).sum(-1)

                    diffs_eps = self.eps * torch.ones_like(diffs)
                    diffs = torch.max(diffs, diffs_eps)
                    logp_sz = torch.nn.functional.log_softmax(-diffs, dim=2)                    

                    val_probs_rots[idx] = -torch.diagonal(logp_sz, 0, 1, 2).cpu().data.numpy()  # Score = - log P(x)

                val_probs_rots = val_probs_rots.sum(1)
                f1_score, precision, thresh, y_pred, y_true = f_score(val_probs_rots, y_test, ratio) 
                save_scv(dir,self.ds, test_data,np.round(val_probs_rots,3),np.round(thresh,3),y_pred, epoch)
                print("Epoch:", epoch, ", fscore: ", f1_score)
                make_graph(dir,self.ds,epoch,0)
                draw_pr_curve(dir,y_true, y_pred,epoch)
                draw_roc_curve(dir,y_true, y_pred,epoch)
                #plt.plot(error_list)
                #plt.plot(error_list)
                #plt.savefig(dir +"/" + str(epoch) + "error_graph.png")
                #plt.cla()
                #plt.clf()

            print('==========Save Model==========')
            save_model(self.ds,epoch,model,optimizer,er,thresh)
                
        
        print("f1_score",f1_score)
        # loss 그래프
        plt.figure(2)
        plt.plot(error_list)
        plt.savefig(dir + "/error_graph.png")
        #plt.show()
        plt.cla()
        plt.clf()
        return f1_score

