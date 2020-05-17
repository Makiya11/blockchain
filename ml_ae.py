from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten,Dropout
from tensorflow.keras.layers import RepeatVector, Input
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
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

tf.compat.v1.random.set_random_seed(0)

# get command input
def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("-set", "--setting", help="Setting Number.")
    parser.add_argument("-act", "--activation", help="Activation.")
    parser.add_argument("-lay", "--layer", help="Number of hidden layers.")
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

def do_ml(SET_NUM, ACTIVATION, HIDDEN_LAYERS, X_train, X_test):
    
    input_dim = X_train.loc[:,setting[SET_NUM]].shape[1]
    input_layer = Input(shape=(input_dim,))
    
    if ACTIVATION =='tanh':
        ACTIVATION1='tanh'
        ACTIVATION2='tanh'
    elif ACTIVATION =='relu':
        ACTIVATION1='relu'
        ACTIVATION2='relu'
    elif ACTIVATION =='mix':
        ACTIVATION1='tanh'
        ACTIVATION2='relu'
    
    if HIDDEN_LAYERS == 2:
        encoder = Dense(int(input_dim / 4), activation=ACTIVATION1, 
                    activity_regularizer=regularizers.l1(10e-5))(input_layer)
        decoder = Dense(input_dim, activation=ACTIVATION2)(encoder)
    elif HIDDEN_LAYERS == 4:
        encoder = Dense(input_dim/ 2, activation=ACTIVATION1, 
                    activity_regularizer=regularizers.l1(10e-5))(input_layer)
        encoder = Dense(int(input_dim / 4), activation=ACTIVATION2)(encoder)
        decoder = Dense(int(input_dim / 2), activation=ACTIVATION1)(encoder)
        decoder = Dense(input_dim, activation=ACTIVATION2)(decoder)
    elif HIDDEN_LAYERS == 6:
        encoder = Dense(input_dim/ 2, activation=ACTIVATION1, 
                    activity_regularizer=regularizers.l1(10e-5))(input_layer)
        encoder = Dense(int(input_dim / 3), activation=ACTIVATION2)(encoder)
        encoder = Dense(int(input_dim / 4), activation=ACTIVATION1)(encoder)
        decoder = Dense(int(input_dim / 3), activation=ACTIVATION2)(encoder)
        decoder = Dense(int(input_dim / 2), activation=ACTIVATION1)(decoder)
        decoder = Dense(input_dim, activation=ACTIVATION2)(decoder)
    elif HIDDEN_LAYERS == 8:
        encoder = Dense(input_dim/ 2, activation=ACTIVATION1, 
                    activity_regularizer=regularizers.l1(10e-5))(input_layer)
        encoder = Dense(int(input_dim / 3), activation=ACTIVATION2)(encoder)
        encoder = Dense(int(input_dim / 3), activation=ACTIVATION1)(encoder)
        encoder = Dense(int(input_dim / 4), activation=ACTIVATION2)(encoder)
        decoder = Dense(int(input_dim / 3), activation=ACTIVATION1)(encoder)
        decoder = Dense(int(input_dim / 3), activation=ACTIVATION2)(decoder)
        decoder = Dense(int(input_dim / 2), activation=ACTIVATION1)(decoder)
        decoder = Dense(input_dim, activation=ACTIVATION2)(decoder)
        
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    nb_epoch = HIDDEN_LAYERS*25
    batch_size = 100
    autoencoder.compile(optimizer='adam', 
                        loss='mean_squared_error', 
                        metrics=['accuracy'])

    train_start = time.time()
    autoencoder.fit(X_train.loc[:,setting[SET_NUM]].values, X_train.loc[:,setting[SET_NUM]].values,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        verbose=0,
                        )
    train_end = time.time()
    train_time = train_end - train_start
    print(f'Training time: {train_time}')
    
    valid_start = time.time()
    validation = autoencoder.predict(X_train.loc[:,setting[SET_NUM]].values)
    valid_end = time.time()
    valid_time = valid_end - valid_start
    print(f'Validation time: {valid_time}')
    
    
    test_start = time.time()
    predictions = autoencoder.predict(X_test.loc[:,setting[SET_NUM]].values)
    test_end = time.time()
    test_time =test_end - test_start
    print(f'Testing time: {test_time}')
    
    mse_valid = np.mean(np.power(X_train.loc[:,setting[SET_NUM]].values - validation, 2), axis=1)
    mse_pred = np.mean(np.power(X_test.loc[:,setting[SET_NUM]].values - predictions, 2), axis=1)
    
    error_train = pd.DataFrame({'train_reconstruction_error': mse_valid,
                                'true_class': y_train,
                                'train time':train_time,
                                'validation time': valid_time
                            
                               })
    
    error_test = pd.DataFrame({'test_reconstruction_error': mse_pred,
                              'true_class': y_test,
                              'test time': test_time})
    
    if not os.path.exists('records_ae'):
        os.mkdir('records_ae')
    error_train.to_csv(f'records_ae/Train{FILE_NUM}_error_param{SET_NUM}_{ACTIVATION}_{HIDDEN_LAYERS}layers.csv')
    error_test.to_csv(f'records_ae/Test{FILE_NUM}_error_param{SET_NUM}_{ACTIVATION}_{HIDDEN_LAYERS}layers.csv')

if __name__ == "__main__":
    options = getOptions(sys.argv[1:])
    SET_NUM = int(options.setting)
    ACTIVATION = options.activation
    HIDDEN_LAYERS = int(options.layer)
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
    
    do_ml(SET_NUM, ACTIVATION, HIDDEN_LAYERS, X_train, X_test)
