import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score, auc
import matplotlib.pyplot as plt
import numpy as np


def draw_pr_curve(dir,labels, preds,epoch):
    precision, recall, thresholds = precision_recall_curve(labels, preds)
    save_name = dir +"/" + str(epoch) + "pr_curve_graph.png"
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
    save_name = dir +"/" + str(epoch) + "roc_curve_graph.png"
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

def make_graph(dir,ds,epoch,train_th):

    if ds == 'kdd_demo':
        file_name = dir + "demo_data_result{}.csv".format(epoch)
        test = pd.read_csv(file_name)
        save_name = dir +"/" + str(epoch) + "kdd_demo_graph.png"
    elif ds == 'cn7_demo':
        file_name = dir + "demo_data_result{}.csv".format(epoch)
        test = pd.read_csv(file_name)
        save_name = dir +"/" + str(epoch) + "cn7_demo_graph.png"
    elif ds == 'thyroid' or ds == 'cn7':
        file_name = dir + "test_data_result{}.csv".format(epoch)
        test = pd.read_csv(file_name)
        test = test.iloc[-100:,:]
        save_name = dir +"/" + str(epoch) + "graph.png"
    elif ds == 'arrhythmia' :
        file_name = dir + "test_data_result{}.csv".format(epoch)
        test = pd.read_csv(file_name)
        test = test.iloc[100:,:]
        save_name = dir +"/" + str(epoch) + "graph.png"
    else:
        file_name = dir + "test_data_result{}.csv".format(epoch)
        test = pd.read_csv(file_name)
        test = test.iloc[198300:198400,:]
        # test = test.iloc[2000:2100,:]
        save_name = dir +"/" + str(epoch) + "graph.png"


    col_name = test.columns.tolist()
    test.rename(columns={str(col_name[-4]):'true', str(col_name[-3]):'pred', str(col_name[-2]):'score', str(col_name[-1]):'threshold'}, inplace=True)
    
    th = test['threshold'].iloc[0]
    print("th",th)

    # 실제값 
    test['color'] = np.where(test.iloc[:,[-4]] == 1, 
                            'red', 
                    np.where(test.iloc[:,[-4]] == 0, 
                            'blue', 
                            'black')
                            )

    # id
    test["id"] = np.arange(len(test.iloc[:,[-2]]))

    fn = 0
    fp = 0
    true_normal = 0
    true_abnormal = 0
    pre_normal = 0
    pre_abnormal = 0

    for true_sample in list(test["true"]):
        if true_sample == 0.0:
             true_normal += 1
        else :
             true_abnormal += 1   

    print('true_normal : ',true_normal)
    print('true_abnormal : ',true_abnormal)

    a_list = list(test["true"] - test["pred"])
    for acc in a_list:
        if acc == -1.0:         # True 인데 False라고 한경우
            fn += 1
        elif acc == 1.0:        # Fals 인데 True라고 한경우
            fp += 1    
        
    count = fn + fp

    count2 = 0
    for k in list(test["pred"]):
        if k == 1:
            count2 += 1


    acc = round((len(a_list)- count) / len(a_list),4)
    print("acc",acc)
    msg = "data_num : {}".format(len(a_list))
    msg1 = "T_normal : {}, T_ano : {} Pre_ano : {}".format(true_normal,true_abnormal,count2)
    msg2 = "false_pred_num : (fn :{} + fp :{} =){}".format(fn,fp,count)
    print(msg)
    print(msg1)
    print(msg2)
    count1 = 0
    for sc in list(test['score']):
        if sc > th:
            count1 +=1 

    print(count1)

    # plt.figure(1)
    # scatter plot
    # test.plot(kind='scatter',
    #         x='id', 
    #         y='score', 
    #         s=5, # marker size
    #         c=test['color']) # marker color by group

    plt.scatter(test['id'],test['score'],c=test['color'])
    
    y_min = th - th*0.5
    y_max = th*10
    plt.axhline(y=th, color='violet', linewidth=2, label='threshold:'+str(th))
    # if train_th != 0:
    #     plt.axhline(y=train_th, color='green', linewidth=2)
    plt.title('Anomaly Score', fontsize=20)
    plt.xlabel('Data id', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    # plt.ylim(1,th + 1000)
    plt.legend(loc='upper left')
    plt.text(0, th , 'acc : '+str(acc))
    plt.text(0, th + y_min*(3/4)*1.5, str(msg))
    plt.text(0, th - y_min*(3/4)*1.7, str(msg1))
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
