import pandas as pd

def load_file(ml, nparam, nfile):
    try:
        df = pd.read_csv(f'records_binary/{ml}_param{nparam}_file{nfile}_binary_classification_report.csv')
    except:
        
        print((f'records_binary/{ml}_param{nparam}_file{nfile}_binary_classification_report.csv'))
    d = {'ML': ml,
         'param': nparam,
         'File': nfile,
         'F1-score': df['f1-score'][3],
         'Accuracy':df['f1-score'][2],
         'train time':df['train time'][0],
         'test time':df['test time'][0]}
    rows.append(d)
rows = []

for ml in ['DNN', 'GB', 'LR', 'RF']:
    for nparam in range(1,5):
        for nfile in range(1,6):
            load_file(ml, nparam, nfile)
df = pd.DataFrame.from_dict(rows)
df.to_csv('supervised.csv', index=False)