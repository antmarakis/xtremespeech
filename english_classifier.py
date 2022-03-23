"""
Basic adjustment from base_classifier.py to only use English data

python english_classifier.py --set_name1 india --set_name2 india
python english_classifier.py --set_name1 india --set_name2 kenya
python english_classifier.py --set_name1 kenya --set_name2 india
python english_classifier.py --set_name1 kenya --set_name2 kenya
"""
import argparse, data_split
import pandas as pd
import numpy as np
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import f1_score, accuracy_score
def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='micro')


parser = argparse.ArgumentParser()
parser.add_argument('--set_name1', required=True)
parser.add_argument('--set_name2', default='default')
parser.add_argument('--danger_label', default=2, type=int)
parser.add_argument('--epochs', default=3, type=int)
parser.add_argument('--model_type', default='bert')
parser.add_argument('--model_name', default='bert-base-cased')
args = parser.parse_args()

TASK = 'extreme' if args.danger_label == 2 else 'acceptable'
DATA_PATH = 'data_splits'

def read_data():
    if ',' in args.set_name1:
        train_data = []
        for name in args.set_name1.split(','):
            train_data.append(pd.read_csv('{}/{}_train.csv'.format(DATA_PATH, name), engine='python'))
        train_df = pd.concat(train_data)
    else:
        train_df = pd.read_csv('{}/{}_train.csv'.format(DATA_PATH, args.set_name1), engine='python')

    if args.set_name2 == 'default':
        if ',' in args.set_name1: print('ERROR! set_name1 must be a list')
        dev_df = pd.read_csv('{}/{}_dev.csv'.format(DATA_PATH, args.set_name1), engine='python')
        test_df = pd.read_csv('{}/{}_test.csv'.format(DATA_PATH, args.set_name1), engine='python')
    elif ',' in args.set_name2:
        dev_data, test_data = [], []
        for name in args.set_name2.split(','):
            dev_data.append(pd.read_csv('{}/{}_dev.csv'.format(DATA_PATH, name), engine='python'))
            test_data.append(pd.read_csv('{}/{}_test.csv'.format(DATA_PATH, name), engine='python'))
        dev_df = pd.concat(dev_data)
        test_df = pd.concat(test_data)
    else:
        dev_df = pd.read_csv('{}/{}_dev.csv'.format(DATA_PATH, args.set_name2), engine='python')
        test_df = pd.read_csv('{}/{}_test.csv'.format(DATA_PATH, args.set_name2), engine='python')

    return clean_data(train_df, dev_df, test_df)


def clean_data(train_df, dev_df, test_df):
    train_df = train_df[pd.notnull(train_df['label'])]
    dev_df = dev_df[pd.notnull(dev_df['label'])]
    test_df = test_df[pd.notnull(test_df['label'])]

    train_df = train_df[pd.notnull(train_df['Text'])].reset_index()
    dev_df = dev_df[pd.notnull(dev_df['Text'])].reset_index()
    test_df = test_df[pd.notnull(test_df['Text'])].reset_index()

    train_df['label'] = train_df['label'].replace({2: args.danger_label}).astype(int)
    dev_df['label'] = dev_df['label'].replace({2: args.danger_label}).astype(int)
    test_df['label'] = test_df['label'].replace({2: args.danger_label}).astype(int)

    train_df['text'] = train_df['text'].astype(str)
    dev_df['text'] = dev_df['text'].astype(str)
    test_df['text'] = test_df['text'].astype(str)
    
    train_df = train_df[train_df['Language'] == 'English']
    dev_df = dev_df[dev_df['Language'] == 'English']
    test_df = test_df[test_df['Language'] == 'English']
    
    print(len(train_df), len(dev_df))

    train_df = train_df[['text', 'label']]
    dev_df = dev_df[['text', 'label']]
    test_df = test_df[['text', 'label']]

    return train_df, dev_df, test_df


def train_model(df):
    model_args = {'manual_seed': 1, 'overwrite_output_dir': True, 'fp16': False,
                  'num_train_epochs': args.epochs, 'output_dir': 'outputs_english_{}'.format(args.set_name1)}
    model = ClassificationModel(args.model_type, args.model_name, num_labels=args.danger_label+1, args=model_args)
    model.train_model(df)
    return model


def make_predictions(model, df):
    result, model_outputs, wrong_predictions = model.eval_model(df, f1=f1_multiclass, acc=accuracy_score)
    return result


def print_predictions(result):
    print(result)
    f = open('res_english_{}_{}_{}_{}.txt'.format(args.set_name1, args.set_name2, TASK, args.model_name), 'a+')
    f.write(str(result['acc']))
    f.write('\n')
    f.close()


def main():
    train_df, dev_df, test_df = read_data()
    model = train_model(train_df)
    
    dev_res = make_predictions(model, dev_df)
    dev0_res = make_predictions(model, dev_df[dev_df['label'] == 0])
    dev1_res = make_predictions(model, dev_df[dev_df['label'] == 1])
    dev2_res = make_predictions(model, dev_df[dev_df['label'] == args.danger_label])

    print_predictions(dev_res)
    print_predictions(dev0_res)
    print_predictions(dev1_res)
    print_predictions(dev2_res)


if __name__ == "__main__":
    main()
