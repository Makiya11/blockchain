from sklearn.svm import OneClassSVM 
from sklearn import preprocessing
from sklearn.metrics import f1_score,accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error 
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import numpy as np
import os
import argparse
import sys
import time
import warnings 

# get command input
def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("-set", "--setting", help="Setting Number.")
    options = parser.parse_args(args)
    return options 

# Importing the training dataset
def load_data(features):
    # Read csv file
    Normal = pd.read_csv('../Normal.csv')
    Abn_Syn = pd.read_csv('../Abn-Syn.csv')
    Abn_1p = pd.read_csv('../Abn-1p.csv')
    Abn_Dos = pd.read_csv('../Abn-DoS.csv')
    
    Normal = Normal.loc[:,features]
    Abn_Syn = Abn_Syn.loc[:,features]
    Abn_1p = Abn_1p.loc[:,features]
    Abn_Dos = Abn_Dos.loc[:,features]

    Normal['label'] = 0
    Abn_Syn['label'] = 1
    Abn_1p['label'] = 2
    Abn_Dos['label'] = 3

    frames = [Normal, Abn_Syn, Abn_1p, Abn_Dos]
    Entire =  pd.concat(frames)
    return Normal, Entire


# feature scalling
def feature_scaling(Normal, Entire):

    label = Entire.pop('label')
    # Normalize 0 to 1
    minmaxscaler = preprocessing.MinMaxScaler(feature_range=(0,1))

    # Combine training file and testing file for normalization       
    Entire_sclaed = pd.DataFrame(minmaxscaler.fit_transform(Entire), columns = Entire.columns[:])
    Entire_sclaed['label'] = label.values
    # Separate training file and testing file
    TRAIN_SIZE = Normal.shape[0]
    Normal_scaled = Entire_sclaed.iloc[:TRAIN_SIZE, :]
    
    return Normal_scaled, Entire_sclaed

def do_ml(SET_NUM, X_train, X_test):

    svm = OneClassSVM(kernel='rbf', gamma='auto')
    train_start = time.time()
    svm.fit(X_train.loc[:,setting[SET_NUM]].values)
    train_end = time.time()
    train_time = train_end - train_start
    print(f'Training time: {train_time}')

    test_start = time.time()
    y_pred = svm.predict(X_test.loc[:,setting[SET_NUM]].values) 
    test_end = time.time()
    test_time =test_end - test_start
    print(f'Testing time: {test_time}')
    
    y_pred = np.where(y_pred==1, 0, 1)
    y_test = np.where(X_test['label']==0, 0, 1)

    df_test = pd.DataFrame({'y_pred':y_pred, 'y_test':y_test}) 
    if not os.path.exists('records_svm'):
        os.mkdir('records_svm')
    df_test.to_csv(f'records_svm/svm_{SET_NUM}.csv')

if __name__ == "__main__":
    options = getOptions(sys.argv[1:])
    SET_NUM = int(options.setting)
    features = ['# VERSION', '# VERACK', '# ADDR', '# INV', '# GETDATA',
                '# GETHEADERS', '# TX', '# HEADERS', '# BLOCK', '# GETADDR',
                '# MEMPOOL', '# PING', '# PONG', '# NOTFOUND', '# SENDHEADERS',
                '# FEEFILTER', '# SENDCMPCT', "# CMPCTBLOCK", '# GETBLOCKTXN',
                '# BLOCKTXN', '# REJECT']

    diff = list(map(lambda x: x+' Diff', features))
    bytesAvg = list(map(lambda x: x[2:]+' BytesAvg Diff', features))
    setting = [0, diff, features+diff, features+diff+bytesAvg]

    Normal, Entire = load_data(setting[SET_NUM])
    Normal, Entire = feature_scaling(Normal, Entire)
    do_ml(SET_NUM, Normal, Entire)