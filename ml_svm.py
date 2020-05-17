from sklearn.svm import OneClassSVM 
from sklearn import preprocessing
from sklearn.metrics import f1_score,accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error 
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
    parser.add_argument("-file", "--file", help="File Number.")
    options = parser.parse_args(args)
    return options 

# Importing the training dataset
def load_data(FILE_NUM):
    
    # Read csv file
    train_dataset = pd.read_csv(f'../Dataset/Train{FILE_NUM}.csv')
    test_dataset = pd.read_csv(f'../Dataset/Test{FILE_NUM}.csv')
    
    train_dataset = train_dataset[train_dataset['label']==0]

    # create Xtrain and ytrain
    X_train = pd.DataFrame(train_dataset.iloc[:, : -1].values, columns = train_dataset.columns[:-1])
    y_train = train_dataset.iloc[:, -1].values
    
    # create Xtest and ytest
    X_test = pd.DataFrame(test_dataset.iloc[:, : -1].values, columns = test_dataset.columns[:-1])
    y_test = test_dataset.iloc[:, -1].values
    
    return X_train, X_test, y_train, y_test 

# feature scalling
def feature_scaling(X_train, X_test):

    # Normalize 0 to 1
    minmaxscaler = preprocessing.MinMaxScaler(feature_range=(0,1))

    # Combine training file and testing file for normalization
    combine = pd.concat([X_train,X_test])        
    combine = pd.DataFrame(minmaxscaler.fit_transform(combine), columns = combine.columns[:])
    
    # Separate training file and testing file
    TRAIN_SIZE = X_train.shape[0]
    X_train = combine.iloc[:TRAIN_SIZE, :]
    X_test = combine.iloc[TRAIN_SIZE:, :]
    
    return X_train, X_test

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

    df_test = pd.DataFrame({'y_pred':y_pred, 'y_test':y_test, 'train time':train_time, 'test time':test_time }) 
    if not os.path.exists('records_svm'):
        os.mkdir('records_svm')
    df_test.to_csv(f'records_svm/file{FILE_NUM}_svm_{SET_NUM}.csv')

if __name__ == "__main__":
    options = getOptions(sys.argv[1:])
    SET_NUM = int(options.setting)
    FILE_NUM = int(options.file)
    
    df_label = pd.read_csv(f'../Dataset/Train{1}.csv', nrows=1)
    all_features = df_label.iloc[:,:-1].columns

    features = ['# VERSION', '# VERACK', '# ADDR', '# INV', '# GETDATA',
                '# GETHEADERS', '# TX', '# HEADERS', '# BLOCK', '# GETADDR',
                '# MEMPOOL', '# PING', '# PONG', '# NOTFOUND', '# SENDHEADERS',
                '# FEEFILTER', '# SENDCMPCT', "# CMPCTBLOCK", '# GETBLOCKTXN',
                '# BLOCKTXN', '# REJECT']

    
    diff = list(map(lambda x: x+' Diff', features))
    bytesAvg = list(map(lambda x: x[2:]+' BytesAvg Diff', features))
    setting = [0, diff, features+diff, features+diff+bytesAvg, all_features]
    
    X_train, X_test, y_train, y_test= load_data(FILE_NUM)
    X_train, X_test = feature_scaling(X_train, X_test)    
    do_ml(SET_NUM, X_train, X_test)
