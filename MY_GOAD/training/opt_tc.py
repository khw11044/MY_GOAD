import torch.utils.data
import numpy as np
import torch
import torch.utils.data
from torch.backends import cudnn
from networks.wideresnet import WideResNet
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

cudnn.benchmark = True

def roc_curve_plot(y_test , pred_proba_c1, epoch, path):
    # 임곗값에 따른 FPR, TPR 값을 반환 받음. 
    fprs , tprs , thresholds = roc_curve(y_test ,pred_proba_c1)
    optimal_idx = np.argmax(tprs  - fprs)
    optimal_threshold = thresholds[optimal_idx]
    y_prob_pred = (pred_proba_c1 >= optimal_threshold).astype(bool)
    print(classification_report(y_test, y_prob_pred, target_names=['abnormal', 'normal']))

    # ROC Curve를 plot 곡선으로 그림. 
    plt.plot(fprs , tprs, label='ROC')
    # 가운데 대각선 직선을 그림. 
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.scatter(fprs[optimal_idx], tprs[optimal_idx], marker='+', s=100, color='r')
  
    # FPR X 축의 Scale을 0.1 단위로 변경, X,Y 축명 설정등   
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('FPR( 1 - Sensitivity )')
    plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.savefig(f'{path}/result.png')
    

def tc_loss(zs, m, device):
    means = zs.mean(0).unsqueeze(0)
    res = ((zs.unsqueeze(2) - means.unsqueeze(1)) ** 2).sum(-1)
    pos = torch.diagonal(res, dim1=1, dim2=2)
    offset = torch.diagflat(torch.ones(zs.size(1))).unsqueeze(0).to(device)  * 1e6
    neg = (res + offset).min(-1)[0]
    loss = torch.clamp(pos + m - neg, min=0).mean()
    return loss


class TransClassifier():
    def __init__(self, num_trans, device, args):
        self.n_trans = num_trans
        self.args = args
        self.device = device
        self.netWRN = WideResNet(self.args.depth, num_trans, self.args.widen_factor).to(self.device)    # .to(self.device) 
        self.optimizer = torch.optim.Adam(self.netWRN.parameters())
        self.save_path = args.save_path

    def fit_trans_classifier(self, x_train, x_test, y_test, class_name):
        print("Training", len(x_train))
        self.netWRN.train()
        bs = self.args.batch_size
        N, sh, sw, nc = x_train.shape
        n_rots = self.n_trans
        m = self.args.m
        celoss = torch.nn.CrossEntropyLoss()
        ndf = 256
        
        error_list = []
        best_auc = 0
        for epoch in range(self.args.epochs):
            rp = np.random.permutation(N//n_rots)
            rp = np.concatenate([np.arange(n_rots) + rp[i]*n_rots for i in range(len(rp))])
            assert len(rp) == N
            all_zs = torch.zeros((len(x_train), ndf)).to(self.device)   # torch.Size([40000, 256])
            diffs_all = []
            total_loss = 0
            n_batches = 0
            
            with tqdm(range(0, len(x_train), bs), desc='Training') as pbar :
                for i in pbar :

                    batch_range = min(bs, len(x_train) - i)
                    idx = np.arange(batch_range) + i
                    xs = torch.from_numpy(x_train[rp[idx]]).float().to(self.device) 
                    zs_tc, zs_ce = self.netWRN(xs)  # ([864, 3, 32, 32]) -> ([864, 256]), ([864, 8])

                    all_zs[idx] = zs_tc
                    train_labels = torch.from_numpy(np.tile(np.arange(n_rots), batch_range//n_rots)).long().to(self.device) 
                    zs = torch.reshape(zs_tc, (batch_range//n_rots, n_rots, ndf))   # ([576, 256]) -> ([72, 8, 256])

                    means = zs.mean(0).unsqueeze(0) # ([72, 8, 256] -> ([1, 8, 256])
                    diffs = -((zs.unsqueeze(2).detach().cpu().numpy() - means.unsqueeze(1).detach().cpu().numpy()) ** 2).sum(-1)
                    diffs_all.append(torch.diagonal(torch.tensor(diffs), dim1=1, dim2=2))

                    tc = tc_loss(zs, m, self.device)
                    ce = celoss(zs_ce, train_labels)
                    if self.args.reg:
                        loss = ce + self.args.lmbda * tc + 10 *(zs*zs).mean()
                    else:
                        loss = ce + self.args.lmbda * tc
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                    n_batches += 1
                    pbar.set_postfix({'loss' : "{}".format(round(loss.item(),7))})

            error_list.append(loss.item())
            print('train  Epoch {}/{}\t Loss: {:.8f}'
                        .format(epoch + 1, self.args.epochs,  total_loss / n_batches))
            
            all_zs = torch.reshape(all_zs, (N//n_rots, n_rots, ndf))        # torch.Size([40000, 256]) -> ([5000, 8, 256])
            normal_means = all_zs.mean(0, keepdim=True)
            
            ### Vaildation
            if epoch%2==0 or (epoch + 1 == self.args.epochs):
                print("Vaildation", len(y_test))

                self.netWRN.eval()
                n_rots_vaild = n_rots # n_rots  # 1   # n_rots
                
                with torch.no_grad():
                    batch_size = bs # int(bs // n_rots) # bs
                    val_probs_rots = np.zeros((len(y_test), n_rots_vaild))      # self.n_trans -> n_rots_vaild
                    for i in tqdm(range(0, len(x_test), batch_size), desc='Vaildating'):
                        batch_range = min(batch_size, len(x_test) - i)  # batch 사이즈 별로 진행하다가 남은 것들 
                        idx = np.arange(batch_range) + i
                        xs = torch.from_numpy(x_test[idx]).float().to(self.device) 

                        zs, fs = self.netWRN(xs)
                        zs = torch.reshape(zs, (batch_range // n_rots_vaild, n_rots_vaild, ndf))    # ([576, 256]) -> ([576, 1, 256])
                        # zs: ([72, 1, 256]) -unsqueeze-> ([72, 1, 1, 256])
                        diffs = ((zs.unsqueeze(2) - normal_means) ** 2).sum(-1)     # normal의 평균 score를 기준으로 진행 ([576, 1, 1]) 
                        diffs_eps = self.args.eps * torch.ones_like(diffs)
                        diffs = torch.max(diffs, diffs_eps)
                        logp_sz = torch.nn.functional.log_softmax(-diffs, dim=2)
                        # logp_sz = torch.nn.functional.log_softmax(-diffs, dim=0)

                        zs_reidx = np.arange(batch_range // n_rots_vaild) + i // n_rots_vaild
                        val_probs_rots[zs_reidx] = -torch.diagonal(logp_sz, 0, 1, 2).cpu().data.numpy() # (576, 1)

                    val_probs_rots = -val_probs_rots.sum(1)  # (10000, 1) -> (10000, )
                    roc_auc = roc_auc_score(y_test, val_probs_rots)
                    fprs , tprs , thresholds = roc_curve(y_test, val_probs_rots)
                    optimal_idx = np.argmax(tprs - fprs)
                    optimal_threshold = thresholds[optimal_idx]
                    y_pred_optimal = val_probs_rots >= optimal_threshold
                    f1 = f1_score(y_test, y_pred_optimal)
                    acc = accuracy_score(y_test, y_pred_optimal)

                    roc_curve_plot(y_test , val_probs_rots, epoch, self.save_path)

                    print('vaild Epoch: {}/{} \t AUC: {:.6f}, f1 score: {:.6f}, acc: {:.6f}, threshold: {:.6f}'
                          .format(epoch + 1, self.args.epochs, roc_auc, f1, acc, optimal_threshold))
                    
                    
                    if best_auc<roc_auc:
                        best_auc = roc_auc
                        print('best vaildation performance: {:.8f} and save model'.format(best_auc))
                        torch.save(
                                    {
                                    'epoch': epoch,
                                    'normal': normal_means,     # normal_means, optimal_threshold
                                    'thresh': optimal_threshold,
                                    'net_dict': self.netWRN.state_dict()
                                    }, 
                                   self.save_path +'/{}_best_model.pth'.format(class_name))

        # loss 그래프
        plt.figure(2)
        plt.plot(error_list)
        plt.savefig(self.save_path + "/error_graph.png")
        #plt.show()
        plt.cla()
        plt.clf()
        return best_auc, normal_means, optimal_threshold

    def test(self, x_test, y_test, normal_means, optimal_threshold):
        print("Test", len(x_test))
        self.netWRN.eval()
        batch_size = self.args.batch_size  # 16
        n_rots = self.n_trans
        n_rots_test = n_rots
        ndf = 256
        
        
        val_probs_rots = np.zeros((len(y_test), n_rots_test))
        with torch.no_grad():
            for i in tqdm(range(0, len(x_test), batch_size), desc='Test'):
                batch_range = min(batch_size, len(x_test) - i)
                idx = np.arange(batch_range) + i
                xs = torch.from_numpy(x_test[idx]).float().to(self.device) 

                zs, fs = self.netWRN(xs)
                zs = torch.reshape(zs, (batch_range // n_rots_test, n_rots_test, ndf))    # ([576, 256])

                diffs = ((zs.unsqueeze(2) - normal_means) ** 2).sum(-1)
                diffs_eps = self.args.eps * torch.ones_like(diffs)
                diffs = torch.max(diffs, diffs_eps)
                logp_sz = torch.nn.functional.log_softmax(-diffs, dim=2)
                # logp_sz = torch.nn.functional.log_softmax(-diffs, dim=0)

                zs_reidx = np.arange(batch_range // n_rots_test) + i // n_rots_test
                val_probs_rots[zs_reidx] = -torch.diagonal(logp_sz, 0, 1, 2).cpu().data.numpy() # (576, 1)
            
            val_probs_rots = -val_probs_rots.sum(1)  # (10000, 1) -> (10000, )
            roc_auc = roc_auc_score(y_test, val_probs_rots)

            y_pred_optimal = val_probs_rots >= optimal_threshold
            f1 = f1_score(y_test, y_pred_optimal)

            print(f"AUC: {roc_auc}, f1 score: {f1}")

            
        return roc_auc, f1