import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import os

def load_file(num, nlayer, act):
    df_train = pd.read_csv(f'records_ae/Train1_error_param{num}_{act}_{nlayer}layers.csv')
    df_test = pd.read_csv(f'records_ae/Test1_error_param{num}_{act}_{nlayer}layers.csv')
    return df_train, df_test

def classification(df_test1, df_test2, df_test3, df_test4, param, act):
    threshold_lst = np.arange(0.005,0.0505,0.005)
    result_list =[[] for i in range(len(threshold_lst))]
    result_list2 =[[] for i in range(len(threshold_lst))]
    cm_list =[[] for i in range(len(threshold_lst))]

    for i, threshold in enumerate(threshold_lst):
        for df in [df_test1, df_test2, df_test3, df_test4]:
            y_pred = [1 if e > threshold_lst[i] else 0 for e in df['test_reconstruction_error']]
            y_test = np.where(df['true_class']==0, 0, 1)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            result_list[i].append(acc)
            result_list2[i].append(f1)
            cm = confusion_matrix(df['true_class'], y_pred)
            tn = cm[0][0]
            tp = cm[1][1]
            fn = cm[1][0]
            fp = cm[0][1]
            cm_list[i].append([tn,fp,fn,tp])
    column_name=['2 Layers', '4 Layers', '6 Layers', '8 Layers']        
    ax = pd.DataFrame(result_list, columns=column_name)
    ax.index = threshold_lst
    ax.plot(style=['s-', 'o-', '^-', '*-', 'D-'],rot=0)

    plt.title(f'param{param} {act}', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlabel('Threshold', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0,1.1)
    plt.legend(fontsize=14)
    plt.savefig(f'fig/Accuracy param{param} {act}.png')
    plt.show()
    plt.close()
    
    ax2 = pd.DataFrame(result_list2, columns=column_name)
    ax2.index = threshold_lst
    ax2.plot(style=['s-', 'o-', '^-', '*-', 'D-'],rot=0)

    plt.title(f'param{param}  {act}', fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    plt.xlabel('Threshold', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0,1.1)
    plt.legend(fontsize=14)
    plt.savefig(f'fig/F1score param{param} {act}.png')
    plt.show()
    plt.close()
if not os.path.exists('fig'):
    os.mkdir('fig')
    
    
for act in ['mix', 'relu', 'tanh']:
    for i in range(1,5):
        df_train1, df_test1 = load_file(i, 2, act)
        df_train2, df_test2 = load_file(i, 4, act)
        df_train3, df_test3 = load_file(i, 6, act)
        df_train4, df_test4 = load_file(i, 8, act)
        classification(df_test1, df_test2, df_test3, df_test4, i, act)
