# 데이터가  Transfer 되지않은 데이터 [(7365,23)]
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import aenet as AEmodel
from sklearn.metrics import precision_recall_fscore_support as prf
import pandas as pd
from graph import make_graph, draw_pr_curve, draw_roc_curve
import os
import matplotlib.pyplot as plt


# AutoEncoder로 구현

def tc_loss(zs, m):
    means = zs.mean(0).unsqueeze(0)
    res = ((zs.unsqueeze(2) - means.unsqueeze(1)) ** 2).sum(-1)
    pos = torch.diagonal(res, dim1=1, dim2=2)
    offset = torch.diagflat(torch.ones(zs.size(1))).unsqueeze(0).cuda() * 1e6
    neg = (res + offset).min(-1)[0]
    loss = torch.clamp(pos + m - neg, min=0).mean()
    return loss

def f_score(scores, labels, ratio):                         # 이게 아웃풋이자 평가 확인 
    # thresh = np.percentile(scores, ratio)
    thresh = (min(scores[np.where(labels == 1)]) - np.mean(scores[np.where(labels != 1)])) /2 
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
    def __init__(self, args, d):
        self.ds = args.dataset
        self.d = d
        self.m = args.m
        self.lmbda = args.lmbda
        self.batch_size = args.batch_size
        self.ndf = args.ndf
        self.n_rots = args.n_rots
        self.d_out = args.d_out
        self.eps = args.eps

        self.n_epoch = args.n_epoch

       
        self.aeNet = AEmodel.autoencoder(self.d ).cuda()
        self.optimizerAE = optim.Adam(self.aeNet.parameters(), lr=args.lr, weight_decay=1e-5)

   
    def fit_trans_classifier(self, train_xs, x_test, y_test, ratio, test_data):
        labels = torch.arange(self.n_rots).unsqueeze(0).expand((self.batch_size, self.n_rots)).long().cuda()
        print('Training')
        error_list = []

        model = self.aeNet
        criterion = nn.MSELoss()
        optimizer = self.optimizerAE

        dir = "checkpoints/" + self.ds + "_checkpoint/"
        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")
        if not os.path.exists(dir):
            os.mkdir(dir)

        for epoch in range(self.n_epoch):
            model.train()
            rp = np.random.permutation(len(train_xs))
            n_batch = 0
            sum_zs = torch.zeros((self.ndf, self.n_rots)).cuda()

            for i in range(0, len(train_xs), self.batch_size):
                # model.zero_grad()
                batch_range = min(self.batch_size, len(train_xs) - i)
                train_labels = labels
                if batch_range == len(train_xs) - i:
                    train_labels = torch.arange(self.n_rots).unsqueeze(0).expand((len(train_xs) - i, self.n_rots)).long().cuda()
                idx = np.arange(batch_range) + i
                xs = torch.from_numpy(train_xs[rp[idx]]).float().cuda()
                # xs = xs.view(xs.size(0), -1)
                output = model(xs)
                loss = criterion(output, xs)
                optimizer.zero_grad()
                loss.backward()


                optimizer.step()
                n_batch += 1

                print('===> Epoch [{}] ({} / {}) : loss : {:.5}'.format(epoch,i, len(train_xs), loss.item()))     # loss도 나중에 저장합시다.
                error_list.append(loss.item())

            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, self.n_epoch, loss.item()))
            if epoch % 10 == 0:     # self.batch_size
                print('===> Epoch [{}] ({} / {}) : loss : {:.5}'.format(epoch,i, len(train_xs), loss.item()))     # loss도 나중에 저장합시다.
                

        #     means = sum_zs.t() / n_batch
        #     means = means.unsqueeze(0)

            

            # ----------------------------------------------평가 ----------------------------------------------
            print('==========Evaluation==========')
            model.eval()

            with torch.no_grad():
                reconstruction_error = np.zeros((len(y_test)))
                for i in range(0, len(x_test), self.batch_size):
                    batch_range = min(self.batch_size, len(x_test) - i)
                    idx = np.arange(batch_range) + i
                    xs = torch.from_numpy(x_test[idx]).float().cuda()
                    output= model(xs)
                    reconstruction_error[idx]  = torch.mean(((xs - output)**2),1).cpu().numpy()


                f1_score, precision, thresh, y_pred, y_true = f_score(reconstruction_error, y_test, ratio) 
                print("f1_score",f1_score)
                print("precision",precision)
                print("thresh",thresh)
                print("y_pred",y_pred)
                print("y_true",y_true)
                print()
                save_scv(dir,self.ds, test_data,np.round(reconstruction_error,3),np.round(thresh,3),y_pred, epoch)
                print("Epoch:", epoch, ", fscore: ", f1_score, ", thresh",thresh)
                make_graph(dir,self.ds,epoch,0)
                draw_pr_curve(dir,y_true, y_pred,epoch)
                draw_roc_curve(dir,y_true, y_pred,epoch)


            print('==========Save Model==========')
            save_model(self.ds,epoch,model,optimizer,loss,thresh)
                
        
        print("f1_score",f1_score)
        # loss 그래프
        plt.figure(2)
        plt.plot(error_list)
        plt.savefig(dir + "/error_graph.png")
        #plt.show()
        plt.cla()
        plt.clf()
        return f1_score

