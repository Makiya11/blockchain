import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,accuracy_score, confusion_matrix, classification_report
import os

def load_file(num, nlayer, act):
    df_train = pd.read_csv(f'records_ae/Train1_error_param{num}_{act}_{nlayer}layers.csv')
    df_test = pd.read_csv(f'records_ae/Test1_error_param{num}_{act}_{nlayer}layers.csv')
    return df_train, df_test

def ecdf(data):
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return(x,y)   

def cdf_plot(df_test1, df_test2, df_test3, df_test4, df_train1, df_train2, df_train3, df_train4, nlayer, act):

    for i,df in enumerate(zip([df_test1, df_test2, df_test3, df_test4],[df_train1, df_train2, df_train3, df_train4])):           
            normal = df[0][df[0]['true_class']==0]['test_reconstruction_error']
            sym = df[0][df[0]['true_class']==1]['test_reconstruction_error']
            onep = df[0][df[0]['true_class']==2]['test_reconstruction_error']
            dos = df[0][df[0]['true_class']==3]['test_reconstruction_error']
            normal_valid = df[1][df[1]['true_class']==0]['train_reconstruction_error']
            
            x1,y1 = ecdf(normal)
            x2,y2 = ecdf(sym)
            x3,y3 = ecdf(onep)
            x4,y4 = ecdf(dos)
            x5,y5 = ecdf(normal_valid)
                          
            plt.plot(x1,y1, label = 'Normal')
            plt.plot(x2,y2, label = 'Abn-Syn')
            plt.plot(x3,y3, label = 'Abn-1p')
            plt.plot(x4,y4, label = 'Abn-Dos')
            plt.plot(x5,y5, label = 'Normal(Validation)', linestyle='dashed',linewidth=2)

            plt.title(f'Reconstruction error Param{i+1} {nlayer}layers {act}',fontsize=14)
            plt.xlabel('Reconstruction error', fontsize=14)
            plt.ylabel('CDF', fontsize=14)
            plt.tick_params(labelsize=14)
            plt.legend(fontsize=14, loc="best")
            plt.savefig(f'fig/Reconstruction error Param{i+1} {nlayer}layers {act}.png')
            plt.show()
            plt.close()
if not os.path.exists('fig'):
    os.mkdir('fig')

for act in ['mix','relu','tanh']:
    for nlayer in range(2,10,2):
        df_train1, df_test1 = load_file(1, nlayer, act)
        df_train2, df_test2 = load_file(2, nlayer, act)
        df_train3, df_test3 = load_file(3, nlayer, act)
        df_train4, df_test4 = load_file(4, nlayer, act)
        cdf_plot(df_test1, df_test2, df_test3, df_test4, df_train1, df_train2, df_train3, df_train4, nlayer, act)