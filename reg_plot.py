import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('regression_top10_WO_NumPeers.csv')

for l in range(len(label)):
    plt.plot(x, df[(df['Classifier']==ml[0]) & (df['ratio']=='50_50')][label[l]].values, label = f'{ml[0]}')
    plt.plot(x, df[(df['Classifier']==ml[1]) & (df['ratio']=='50_50')][label[l]].values, label = f'{ml[1]}')
    plt.plot(x, df[(df['Classifier']==ml[2]) & (df['ratio']=='50_50')][label[l]].values, label = f'{ml[2]}')
    plt.tick_params(labelsize=14)
    plt.xticks(x)
    plt.xlim(max(x), min(x))
    plt.ylabel(y_axis[l], fontsize=15)
    plt.xlabel('Number of features', fontsize=15)
    plt.legend(fontsize=15)
    plt.show()
    plt.close()
    
df2 = pd.read_csv('regression_feature_comparison.csv')

df_1 = df2[df2['feature_set']==1].reset_index(drop=True)
df_2 = df2[df2['feature_set']==2].reset_index(drop=True) 
df_3 = df2[df2['feature_set']==3].reset_index(drop=True) 
df_4 = df2[df2['feature_set']==4].reset_index(drop=True) 
df_1.rename(columns=lambda s: s+'_1', inplace =True)
df_2.rename(columns=lambda s: s+'_2', inplace =True)
df_3.rename(columns=lambda s: s+'_3', inplace =True)
df_4.rename(columns=lambda s: s+'_4', inplace =True)

df_5 = pd.concat([df_1,df_2, df_3, df_4], axis=1)
df_reverse = df_5.iloc[::-1].reset_index(drop=True)

RMSE = ['RMSE_3', 'RMSE_4','RMSE_1', 'RMSE_2']
TRAIN_TIME = ['train_time_3', 'train_time_4','train_time_1', 'train_time_2']
TEST_TIME = ['test_time_3', 'test_time_4','test_time_1', 'test_time_2']
df_reverse.index = ['LR (S-1)', 'LR (S-2)', 'LR (S-3)', 'RF (S-1)', 'RF (S-2)', 'RF (S-3)','GB (S-1)', 'GB (S-2)', 'GB (S-3)' ]

rf = df_reverse.loc[['RF (S-1)', 'RF (S-2)', 'RF (S-3)'],:]
df_reverse = df_reverse.drop(['RF (S-1)', 'RF (S-2)', 'RF (S-3)'],axis=0)
df_vis = pd.concat([df_reverse,rf])

df_vis[RMSE].plot.bar(rot=30)
plt.tick_params(labelsize=15)
plt.ylabel('RMSE', fontsize=15)
plt.ylim(0,1)
plt.legend(['Entire features', 'Entire features w/o NumPeers', 'top10 features', 'top10 features w/o NumPeers'], fontsize=13)
plt.show()
plt.close()

df_vis[TRAIN_TIME].plot.bar(rot=30)
plt.tick_params(labelsize=15)
plt.legend(['Entire features', 'Entire features w/o NumPeers', 'Top10 features', 'Top10 features w/o NumPeers'], fontsize=12)
plt.ylabel('Training time (sec)', fontsize=15)
plt.show()
plt.close()


df_vis[TEST_TIME].plot.bar(rot=30)
plt.tick_params(labelsize=15)
plt.legend(['Entire features', 'Entire features w/o NumPeers', 'Top10 features', 'Top10 features w/o NumPeers'], fontsize=13)
plt.ylabel('Testing time (sec)', fontsize=15)
plt.show()
plt.close()