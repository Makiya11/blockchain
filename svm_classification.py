import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,accuracy_score, confusion_matrix, classification_report

def load_file(num):
    df = pd.read_csv(f'records_svm/svm_{num}.csv')

    cr = classification_report(df['y_test'], df['y_pred'], digits=3, output_dict=True)
    accuracy = accuracy_score(df['y_test'], df['y_pred'])
    cr['macro avg']['accuracy']=accuracy
    
    return cr['macro avg']

cr1 = load_file(1)
cr2 = load_file(2)
cr3 = load_file(3)

result = pd.DataFrame([cr1, cr2,cr3], index = ['param1','param2','param3'])
result.drop(['support'], axis=1, inplace=True)
result.to_csv('svm_classification_result.csv')