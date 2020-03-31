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

tf.compat.v1.random.set_random_seed(0)

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
    input_dim = X_train.loc[:,setting[SET_NUM]].shape[1]
    encoding_dim = len(setting[SET_NUM])
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim/ 2, activation="tanh", 
                    activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoder = Dense(int(encoding_dim / 4), activation="relu")(encoder)
    decoder = Dense(int(encoding_dim / 4), activation='tanh')(encoder)
    decoder = Dense(input_dim, activation='relu')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)

    nb_epoch = 100
    batch_size = 32
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
    
    test_start = time.time()
    predictions = autoencoder.predict(X_test.loc[:,setting[SET_NUM]].values)
    test_end = time.time()
    test_time =test_end - test_start
    print(f'Testing time: {test_time}')
    
    mse = np.mean(np.power(X_test.loc[:,setting[SET_NUM]].values - predictions, 2), axis=1)
    
    error_df = pd.DataFrame({'reconstruction_error': mse,
                            'true_class': X_test['label'].values})
    
    if not os.path.exists('records_ae'):
        os.mkdir('records_ae')

    error_df.to_csv(f'records_ae/error_{SET_NUM}.csv')

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
