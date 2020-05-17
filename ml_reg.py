import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.utils import shuffle
import os
import argparse
import sys
import psutil
import time

# get command input
def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("-cls", "--classifier", help="Machine learning classifier.")
    parser.add_argument("-rat", "--ratio", help="Ratio.")
    parser.add_argument("-set", "--sets", help="Set of features.")
    options = parser.parse_args(args)
    return options

# Importing the training dataset
def load_data(ratio_num, FEATURE_SET):

    dataset = pd.read_csv('connection_dataset.csv')
    dataset.drop(["Timestamp", "Timestamp (Seconds)","Unnamed: 278","Connections"], axis = 1, inplace = True)

    dataset = shuffle(dataset, random_state=0)

    X = dataset.loc[:, feature_lst[FEATURE_SET]]
    y = dataset.iloc[:, -1].values

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range = (0, 1))

    X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns[:], index=X.index)
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = ratio_num, shuffle=False)

    return X_train, X_test, y_train, y_test

def do_ml(X_train, X_test, classifier, CLF, RATIO, FEATURE_SET):
    train_start = time.time()
    regressor = classifier.fit(X_train.values, y_train)
    train_end = time.time()
    train_time = train_end - train_start

    # Predicting the Test set results
    test_start = time.time()
    y_pred = regressor.predict(X_test.values)
    test_end = time.time()
    test_time = test_end - test_start

    rmse = sqrt(mean_squared_error(y_test, y_pred))
    print(f'{CLF} {RATIO} RMSE is {rmse}, train time{train_time}, test time{test_time}')
    dic = {'y_pred': y_pred, 'y_test': y_test}
    df = pd.DataFrame(dic)
    df.to_csv(f'records_reg/{CLF}_{RATIO}_{FEATURE_SET}set_comparison.csv')

    save_df = pd.read_csv('regression_feature_comparison.csv')
    d = {'Classifier':CLF,'ratio':RATIO, 'train_time':train_time,
            'test_time':test_time, 'RMSE':rmse, 'feature_set':FEATURE_SET}
    save_df = save_df.append(d, ignore_index=True)
    save_df.to_csv('regression_feature_comparison.csv',index=False)

if __name__ == "__main__":
    options = getOptions(sys.argv[1:])
    CLF = options.classifier
    RATIO = options.ratio
    FEATURE_SET = int(options.sets)
    
    if not os.path.exists('records_reg'):
        os.mkdir('records_reg') 
    if not os.path.exists('regression_feature_comparison.csv'):
        save_df = pd.DataFrame(columns=['Classifier','ratio','train_time','test_time','RMSE','feature_set'])
        save_df.to_csv('regression_feature_comparison.csv',index=False)

    if RATIO == '20_80':
        ratio_num = 4/5
    elif RATIO == '50_50':
        ratio_num = 1/2
    elif RATIO == '80_20':
        ratio_num = 1/5
        
    top10 = ['NumPeers', 'VERSION BytesMax', '# INV', '# TX', 'INV BytesMax',
   'SENDCMPCT ClocksMax', 'ADDR ClocksAvg', 'FEEFILTER ClocksMax',
   'FEEFILTER ClocksAvg', 'TX BytesAvg']

    top10WOnum = ['VERSION BytesMax', '# VERACK', '# VERSION', '# INV', '# TX',
       'INV BytesMax', 'SENDCMPCT ClocksMax', 'ADDR ClocksAvg',
       'FEEFILTER ClocksMax', 'TX BytesAvg']

    df_label = pd.read_csv(f'../Dataset/connection_dataset.csv', nrows=1)
    df_label.drop(["Timestamp", "Timestamp (Seconds)","Unnamed: 278","Connections", "label"], axis = 1, inplace = True)

    all_features = df_label.columns.to_list()
    all_featuresWOnum = all_features.copy()
    all_featuresWOnum.remove('NumPeers')

    feature_name = ['top10', 'top10_WO_NumPeers', 'all_features', 'all_features_WO_NumPeers']
    feature_lst = [0, top10, top10WOnum, all_features, all_featuresWOnum]


    X_train, X_test, y_train, y_test = load_data(ratio_num, FEATURE_SET)

    if CLF == 'GB':
        from sklearn.ensemble import GradientBoostingRegressor
        classifier = GradientBoostingRegressor(n_estimators=100, random_state = 0)
        do_ml(X_train, X_test, classifier, CLF, RATIO, FEATURE_SET)
    elif CLF == 'RF':
        from sklearn.ensemble import RandomForestRegressor
        classifier = RandomForestRegressor(n_estimators=100, random_state = 0)
        do_ml(X_train, X_test, classifier, CLF, RATIO, FEATURE_SET)
    elif CLF == 'LR':
        from sklearn.linear_model import LinearRegression
        classifier = LinearRegression()
        do_ml(X_train, X_test, classifier, CLF, RATIO, FEATURE_SET)