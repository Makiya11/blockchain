import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,accuracy_score, confusion_matrix, classification_report
import os

def load_file(num):
    df_Test = pd.read_csv(f'records_ae/error_{num}.csv')
    return df_Test

def classification(df):
    
    # 0.01 - 0.99 quantile
    low99 = df[df['true_class']==0]['reconstruction_error'].quantile(0.01)
    high99 = df[df['true_class']==0]['reconstruction_error'].quantile(0.99)
    
    # 0.055 - 0.95 quantile
    low95 = df[df['true_class']==0]['reconstruction_error'].quantile(0.05)
    high95 = df[df['true_class']==0]['reconstruction_error'].quantile(0.95)
    
    y_pred_99 = [0 if low99 <= e <= high99  else 1 for e in df['reconstruction_error'].values]
    y_pred_95 = [0 if low95 <= e <= high95  else 1 for e in df['reconstruction_error'].values]

    y_test = np.where(df['true_class']==0, 0, 1)
    cr99 = classification_report(y_test, y_pred_99, digits=3, output_dict=True)
    cr95 = classification_report(y_test, y_pred_95, digits=3, output_dict=True)
    
    accuracy99 = accuracy_score(y_test, y_pred_99)
    accuracy95 = accuracy_score(y_test, y_pred_95)
    
    cr99['macro avg']['accuracy']=accuracy99
    cr95['macro avg']['accuracy']=accuracy95
    
    return cr99['macro avg'], cr95['macro avg']

def ecdf(data):
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return(x,y)   

def plot(df_1, df_2, df_3):
    one = df_1[df_1['true_class']==0]['reconstruction_error']
    two = df_2[df_2['true_class']==0]['reconstruction_error']
    three = df_3[df_3['true_class']==0]['reconstruction_error']
    
    x1,y1 = ecdf(one)
    x2,y2 = ecdf(two)
    x3,y3 = ecdf(three)

    plt.plot(x1, y1, label = 'param1')
    plt.plot(x2, y2, label = 'param2')
    plt.plot(x3, y3, label = 'param3')
    plt.xlabel('Reconstruction error', fontsize=14)
    plt.ylabel('CDF', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.legend(fontsize=14, loc="best")
    plt.title('Profiling of normal instances',fontsize=14)
    plt.savefig('fig/cdf_normal.png')
    plt.show()
    plt.close()

    for i,df in enumerate([df_1, df_2, df_3]):
            normal = df[df['true_class']==0]['reconstruction_error']
            sym = df[df['true_class']==1]['reconstruction_error']
            onep = df[df['true_class']==2]['reconstruction_error']
            dos = df[df['true_class']==3]['reconstruction_error']
            
            x4,y4 = ecdf(normal)
            x5,y5 = ecdf(sym)
            x6,y6 = ecdf(onep)
            x7,y7 = ecdf(dos)

            plt.plot(x4,y4, label = 'Normal')
            plt.plot(x5,y5, label = 'Abn-1p')
            plt.plot(x6,y6, label = 'Abn-DoS')
            plt.plot(x7,y7, label = 'Abn-Syn')

            plt.title(f'Reconstruction error Param{i+1}',fontsize=14)
            plt.xlabel('Reconstruction error', fontsize=14)
            plt.ylabel('CDF', fontsize=14)
            plt.tick_params(labelsize=14)
            plt.legend(fontsize=14, loc="best")
            plt.savefig(f'fig/cdf_param{i+1}.png')
            plt.show()
            plt.close()
    ax = pd.DataFrame({'param1':df_1[df_1['true_class']==0]['reconstruction_error'],
              'param2':df_2[df_2['true_class']==0]['reconstruction_error'],
              'param3':df_3[df_3['true_class']==0]['reconstruction_error']} ).plot.box(showfliers=False)
    ax.set_title('Profiling of normal instances',fontsize=14)
    ax.set_ylabel('Reconstruction Error',fontsize=14)
    ax.tick_params(labelsize=14)
    fig = plt.gcf()
    fig.savefig('fig/box_plot.png')
    plt.show()
    plt.close()

df_1 = load_file(1)
df_2 = load_file(2)
df_3 = load_file(3)

cr99_one, cr95_one = classification(df_1)
cr99_two, cr95_two = classification(df_2)
cr99_three, cr95_three = classification(df_3)

if not os.path.exists('fig'):
        os.mkdir('fig')
plot(df_1, df_2, df_3)

result = pd.DataFrame([cr99_one, cr99_two,cr99_three,cr95_one, cr95_two, cr95_three],
            index = ['param1(99%)','param2(99%)','param3(99%)','param1(95%)','param2(95%)','param3(95%)'])
result.drop(['support'], axis=1, inplace=True)
result.to_csv('ae_classification_result.csv')