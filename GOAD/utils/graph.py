import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score, auc




def draw_pr_curve(dir,labels, preds,epoch):
    precision, recall, thresholds = precision_recall_curve(labels, preds)
    save_name = dir +"/" + str(epoch) + "_pr_curve_graph.png"
    plt.plot([0, 1], [0.5, 0.5], '--')
    plt.plot(recall, precision, label='Logistic')
    plt.legend()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(save_name)
    # plt.ylim(-0.05, 1.05)
    # plt.show()
    plt.cla()
    plt.clf()


def draw_roc_curve(dir,labels, preds,epoch):
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = roc_auc_score(labels, preds)
    save_name = dir +"/" + str(epoch) + "_roc_curve_graph.png"
    plt.title('ROC AUC: {0:.4f}'.format(roc_auc))
    plt.plot([0, 1], [0, 1], '--', label='Random')
    plt.plot(fpr, tpr, label='ROC')
    plt.xlabel('FPR ( 1 - Sensitivity )')
    plt.ylabel('TPR ( Recall )')
    plt.legend()
    plt.savefig(save_name)
    # plt.show()
    plt.cla()
    plt.clf()

def make_graph(dir,ds,epoch,f1_score):

    file_name = dir + "val_data_result_{}.csv".format(epoch)
    test = pd.read_csv(file_name)
    save_name = dir +"/" + str(epoch) + "_val_result_vis.png"

    # col_name = test.columns.tolist()
    # test.rename(columns={str(col_name[-4]):'true', str(col_name[-3]):'pred', str(col_name[-2]):'score', str(col_name[-1]):'threshold'}, inplace=True)
    
    th = test['threshold'].iloc[0]

    # 실제값 
    test['color'] = np.where(test.iloc[:,[-4]] == 1,    # abnorm
                            'red', 
                    np.where(test.iloc[:,[-4]] == 0,    # normal
                            'blue', 
                            'black')
                            )

    # id
    test["id"] = np.arange(len(test))

    fn = 0  # 
    fp = 0
    true_normal = len(test[test["true"]==0])
    true_abnormal = len(test[test["true"]==1])
    pre_normal = 0
    pre_abnormal = 0


    a_list = list(test["true"] - test["pred"])
    
    fn = a_list.count(-1)       # True 인데 False라고 한경우
    fp = a_list.count(1)        # Fals 인데 True라고 한경우 
        
    count = fn + fp

    count2 = len(test[test["pred"]==1])

    acc = round((len(a_list)- count) / len(a_list), 4)
    count1 = len(test[test["score"]>th])
    
    msg = f"true_normal : {true_normal}, true_abnormal : {true_abnormal} = Test_data_num : {len(a_list)}"
    msg1 = "T_normal : {}, T_ano : {}, Pre_ano : {}".format(true_normal,true_abnormal,count2)
    msg2 = "false_pred_num : (fn :{} + fp :{} =){}".format(fn,fp,count)
    print(f"{msg} // {msg2} // collect count: {len(a_list) - count}")

    plt.scatter(test['id'],test['score'],c=test['color'])       # x값, y값
    
    y_min = th + th*0.2
    y_max = th*10
    plt.ylim([0, y_max])     # Y축의 범위: [ymin, ymax]
    
    plt.axhline(y=th, color='violet', linewidth=2, label='threshold:'+str(th))
    # if train_th != 0:
    #     plt.axhline(y=train_th, color='green', linewidth=2)
    plt.title('Anomaly Score', fontsize=20)
    plt.xlabel('Data id', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    # plt.ylim(1,th + 1000)
    plt.legend(loc='upper left')
    plt.text(0, y_max*0.85, str(msg))
    plt.text(0, y_max*0.8, str(msg2))
    plt.text(0, y_max*0.75 , 'acc : '+str(acc))
    plt.text(0, y_max*0.7 , 'total val dataset f1 score : '+str(f1_score))
#     plt.text(0, y_max - y_min*(3/4)*1.9, str(msg2))
    plt.savefig(save_name)
    plt.cla()
    plt.clf()
#     if visual == True:
    # plt.show()
    # plt.cla()
    # plt.clf()

    # plt.figure(2)
    # plt.plot(error_list)
    # plt.savefig(dir +"/" + str(epoch) + "error_graph.png")
    # #plt.show()
    # plt.cla()
    # plt.clf()
