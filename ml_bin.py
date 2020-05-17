import argparse
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn import preprocessing
import numpy as np
import os
import sys
import time

# get command input
def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("-cls", "--classifier", help="Machine learning classifier.")
    parser.add_argument("-set", "--setting", help="Setting Number.")
    parser.add_argument("-file", "--file", help="File Number.")
    options = parser.parse_args(args)
    return options  

# Importing the training dataset
def load_data(FILE_NUM):
    
    # Read csv file
    train_dataset = pd.read_csv(f'../Dataset/Train{FILE_NUM}.csv')
    test_dataset = pd.read_csv(f'../Dataset/Test{FILE_NUM}.csv')

    # create Xtrain and ytrain
    X_train = pd.DataFrame(train_dataset.iloc[:, : -1].values, columns = train_dataset.columns[:-1])
    y_train = train_dataset.iloc[:, -1].values
    y_train = np.where(y_train==0, 0, 1)
    
    # create Xtest and ytest
    X_test = pd.DataFrame(test_dataset.iloc[:, : -1].values, columns = test_dataset.columns[:-1])
    y_test = test_dataset.iloc[:, -1].values
    y_test = np.where(y_test==0, 0, 1)
    
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

# Deep Neural Network
def create_network(input_dim):
    model = Sequential()
    model.add(Dense(50, activation = 'relu', input_dim = input_dim))
    model.add(Dropout(0.2))

    model.add(Dense(50, activation = 'relu'))
    model.add(Dropout(0.2))

    model.add(Dense(50, activation = 'relu'))
    model.add(Dropout(0.2))

    model.add(Dense(50, activation = 'relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1,activation = 'sigmoid'))
    
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

def do_ml(X_train, X_test, classifier, set_num):
    
    train_start = time.time()
    classifier.fit(X_train.loc[:,setting[set_num]].values, y_train)
    train_end = time.time()
    train_time = train_end - train_start
    print(f'Training time: {train_time}')
    
    test_start = time.time()
    y_pred = classifier.predict(X_test.loc[:,setting[set_num]].values)
    test_end = time.time()
    test_time =test_end - test_start
    print(f'Testing time: {test_time}')
    
    y_pred  = y_pred.flatten()
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='micro')
    cm2 = classification_report(y_test, y_pred, digits=3, output_dict=True)

    df_report = pd.DataFrame(cm2).transpose()
#     acr = cm.diagonal()/cm.sum(axis=1)
#     acr = np.append(acr,[0,0,0])
#     df_report['accuracy'] = accuracy_score(y_test, y_pred)
    df_report['train time'] = train_time
    df_report['test time'] = test_time
    df_confusion = pd.crosstab(y_test, y_pred)
    return df_report, df_confusion

if __name__ == "__main__":

    options = getOptions(sys.argv[1:])
    CLF = options.classifier
    SET_NUM = int(options.setting)
    FILE_NUM = int(options.file)

    X_train, X_test, y_train, y_test= load_data(FILE_NUM)
    X_train, X_test = feature_scaling(X_train, X_test)
    
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

    if CLF == 'GB':
        from sklearn.ensemble import GradientBoostingClassifier 
        classifier = GradientBoostingClassifier(n_estimators=100, random_state = 0)
        df_report, df_confusion = do_ml(X_train, X_test, classifier, SET_NUM)
    elif CLF == 'RF':
        from sklearn.ensemble import  RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators=100, random_state = 0)
        df_report, df_confusion = do_ml(X_train, X_test, classifier, SET_NUM)
    elif CLF == 'LR':
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(solver='liblinear', random_state = 0)
        df_report, df_confusion = do_ml(X_train, X_test, classifier, SET_NUM)
    elif CLF == 'SVM':
        from sklearn.svm import SVC
        classifier = SVC(gamma='auto',random_state = 0)
        df_report, df_confusion = do_ml(X_train, X_test, classifier, SET_NUM)
    elif CLF == 'DNN':
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
        classifier = KerasClassifier(build_fn=create_network, input_dim=len(setting[SET_NUM]), epochs=50, batch_size = 100, verbose=0)
        df_report, df_confusion = do_ml(X_train, X_test, classifier, SET_NUM)

    if not os.path.exists('records_binary'):
        os.mkdir('records_binary') 
    df_report.to_csv(f'records_binary/{CLF}_param{SET_NUM}_file{FILE_NUM}_binary_classification_report.csv')
    df_confusion.to_csv(f'records_binary/{CLF}_param{SET_NUM}_file{FILE_NUM}_binary_cm.csv')
    # print(df_report)