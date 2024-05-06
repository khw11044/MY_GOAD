import os 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networks.aenet as AEmodel
from sklearn.metrics import precision_recall_fscore_support as prf
from utils.graph import make_graph, draw_pr_curve, draw_roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

def tc_loss(zs, m):
    means = zs.mean(0).unsqueeze(0)
    res = ((zs.unsqueeze(2) - means.unsqueeze(1)) ** 2).sum(-1)
    pos = torch.diagonal(res, dim1=1, dim2=2)
    offset = torch.diagflat(torch.ones(zs.size(1))).unsqueeze(0).cuda() * 1e6
    neg = (res + offset).min(-1)[0]
    loss = torch.clamp(pos + m - neg, min=0).mean()
    return loss

def f_score(scores, labels, ratio):
    thresh = np.percentile(scores, ratio)
    y_pred = (scores >= thresh).astype(int)
    y_true = labels.astype(int)
    precision, recall, f_score, support = prf(y_true, y_pred, average='binary')
    return round(f_score,5), round(precision,5), round(recall,5), thresh, y_pred, y_true

def save_model(ds,epoch,model,optimizer,er,thresh,mus,sds):
    first_folder = "checkpoints"
    folder_name = first_folder + "/" + ds + "_checkpoint_AE/"
    os.makedirs(first_folder, exist_ok=True)
    os.makedirs(folder_name, exist_ok=True)
    
    model_out_path = folder_name + "model_epoch_{}.pth".format(epoch)
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'er': er,
                'thresh': thresh,
                'mus':mus,
                'sds':sds
                },model_out_path)

def save_valid_scv(folder_name, val_data, val_probs_rots, thresh, y_pred, epoch):

    file_out_path = folder_name + "val_data_result_{}.csv".format(epoch)
    y_pred = y_pred.reshape(-1,1)
    scores = val_probs_rots.reshape(-1,1)
    threshs = np.full(y_pred.shape,thresh)
    val_data_result = np.hstack((val_data,y_pred,scores,threshs))
    df = pd.DataFrame(val_data_result)
    col_name = df.columns.tolist()
    df.rename(columns={col_name[-4]:'true', col_name[-3]:'pred', col_name[-2]:'score', col_name[-1]:'threshold'}, inplace=True)
    
    df_abnorm = df[df['true'] == 1][:100]                    # abnorm
    df_norm = df[df['true'] == 0][:100]                    # abnorm
    df_con = pd.concat((df_norm,df_abnorm)).reset_index(drop=True)
    # df_y_test.to_csv(file_out_path, index=None, header=None)
    df_con.to_csv(file_out_path, index=None)
    

class TransClassifierTabular():
    def __init__(self, args, features_len):
        self.ds = args.dataset
        self.m = args.m
        self.lmbda = args.lmbda
        self.batch_size = args.batch_size
        self.ndf = args.ndf
        self.n_rots = args.n_rots
        self.d_out = args.d_out
        self.eps = args.eps
        self.n_epoch = args.n_epoch
        
        self.f = features_len
        
        self.aeNet = AEmodel.autoencoder(self.f ).cuda()
        # self.optimizerC = optim.Adam(self.aeNet.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.optimizerC = optim.Adam(self.aeNet.parameters(), lr=args.lr, weight_decay=1e-5)


    def fit_trans_classifier(self, train_xs, x_val, y_val, val_dataset, ratio, mus, sds, vis=True):
        labels = torch.arange(self.n_rots).unsqueeze(0).expand((self.batch_size, self.n_rots)).long().cuda()
        criterion = nn.MSELoss()
        model = self.aeNet
        optimizer = self.optimizerC
        
        first_folder = "checkpoints"
        dir_name = first_folder + "/" + self.ds + "_checkpoint_AE/"
        os.makedirs(first_folder, exist_ok=True)
        os.makedirs(dir_name, exist_ok=True)
        
        print('Training')
        error_list = []
        best_perform = 0
        for epoch in range(self.n_epoch):
            model.train()
            rp = np.random.permutation(len(train_xs))
            n_batch = 0
            sum_zs = torch.zeros((self.ndf, self.n_rots)).cuda()

            print('==========Training==========')
            with tqdm(range(0, len(train_xs), self.batch_size)) as pbar :
                for i in pbar :
                    model.zero_grad()
                    batch_range = min(self.batch_size, len(train_xs) - i)
                    train_labels = labels
                    if batch_range == len(train_xs) - i:
                        train_labels = torch.arange(self.n_rots).unsqueeze(0).expand((len(train_xs) - i, self.n_rots)).long().cuda()
                    idx = np.arange(batch_range) + i
                    xs = train_xs[rp[idx]]
                    output = model(xs)
                    loss = criterion(output, xs)
                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()
                    n_batch += 1
                    pbar.set_postfix({'loss' : "{}".format(round(loss.item(),7))})

            means = sum_zs.t() / n_batch
            means = means.unsqueeze(0)
            error_list.append(loss.item())
            print('===> Epoch [{}] ({} / {}) : loss : {:.5}'.format(epoch,i, len(train_xs), loss.item()))
            
            print('==========Evaluation==========')
            model.eval()
            with torch.no_grad():
                reconstruction_error = np.zeros((len(y_val)))
                
                for i in tqdm(range(0, len(x_val), self.batch_size)):
                    batch_range = min(self.batch_size, len(x_val) - i)
                    idx = np.arange(batch_range) + i
                    # xs = torch.from_numpy(x_val[idx]).float().cuda()
                    xs = x_val[idx]
                    output = model(xs)       # ([64, 64, 64]) -> ([64, 64, 64])
                    reconstruction_error[idx] = torch.mean(((xs - output)**2),1).cpu().numpy()      # (64,) = (64, 64)


                f1_score, precision, recall, thresh, y_pred, y_true = f_score(reconstruction_error, y_val, ratio)
                print(f"Epoch: {epoch}, fscore: {f1_score}, precision: {precision}, recall: {recall}, threshold: {thresh}")
            
            
            if best_perform<f1_score:
                best_perform = f1_score
                print('Save Model')
                save_model(self.ds, epoch, model, optimizer, loss, thresh, mus, sds)
                
            if vis:
                save_valid_scv(dir_name, val_dataset, reconstruction_error, thresh, y_pred, epoch)
                make_graph(dir_name,self.ds,epoch,f1_score)
                draw_pr_curve(dir_name,y_true, y_pred,epoch)
                draw_roc_curve(dir_name,y_true, y_pred,epoch)
        
        
        print("f1_score",f1_score)
        # loss 그래프
        plt.figure(2)
        plt.plot(error_list)
        plt.savefig(dir_name + "/AE_error_graph.png")
        #plt.show()
        plt.cla()
        plt.clf()
        return f1_score

