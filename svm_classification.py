import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import os

def load_file(num):
    df = pd.read_csv(f'records_svm/svm_{num}.csv')
    return df

def classification(param, df):

    y_test = np.where(df['y_test']==0, 0, 1)
    acc = accuracy_score(y_test, df['y_pred'])
    f1 = f1_score(y_test, df['y_pred'])
    print('accuracy', acc)
    print('f1', f1)
    d = {'param':param,'accuracy':acc, 'F1 score':f1, 'train time':df['train time'][0], 'test time':df['test time'][0]}
    rows.append(d)
rows= []

for i in range(1,5):
    df = load_file(i)
    classification(i, df)
df_svm = pd.DataFrame.from_dict(rows)
df_svm.to_csv('OC-SVM.csv')