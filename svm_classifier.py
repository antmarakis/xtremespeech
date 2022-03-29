"""
python svm_classifier.py --set_name1 kenya --set_type Country
python svm_classifier.py --set_name1 ek --set_type User
"""
import os
import argparse, data_split
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='micro')


parser = argparse.ArgumentParser()
parser.add_argument('--set_name1', required=True)
parser.add_argument('--set_name2', default='default')
parser.add_argument('--set_type', required=True)
parser.add_argument('--danger_label', default=2, type=int)
parser.add_argument('--max_features', default=5000, type=int)
args = parser.parse_args()

TASK = 'extreme' if args.danger_label == 2 else 'acceptable'
DATA_PATH = 'data_splits'
f = open('res_svm_{}_{}.txt'.format(args.set_name1, TASK), 'w')
f.close()

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

    train_df = train_df[['text', 'label']]
    dev_df = dev_df[['text', 'label']]
    test_df = test_df[['text', 'label']]

    return train_df, dev_df, test_df


def format_data(data, max_features, count_vect=None, tf_transformer=None):
    data = data.sample(frac=1).reset_index(drop=True)
    data['text'] = data['text'].apply(lambda x: str(x))

    Y = data['label'].values
    X = data['text']

    if not tf_transformer:
        count_vect = CountVectorizer(max_features=max_features, lowercase=False)
        X = count_vect.fit_transform(X)

        tf_transformer = TfidfTransformer(use_idf=False).fit(X)
        X = tf_transformer.transform(X)
    else:
        X = count_vect.transform(X)
        X = tf_transformer.transform(X)

    return X, Y, count_vect, tf_transformer


train_df, dev_df, test_df = read_data()


def make_and_print_results(model, X, Y):
    preds = model.predict(X)
    acc = accuracy_score(preds, Y)
    print(acc)
    f = open('res_svm_{}_{}.txt'.format(args.set_name1, TASK), 'a+')
    f.write('{}'.format(acc))
    f.write('\n')
    f.close()


def main():
    max_features = args.max_features
    X_train, Y_train, cv, tf = format_data(train_df, max_features)
    X_dev, Y_dev, _, _ = format_data(dev_df, max_features, cv, tf)
    X_dev0, Y_dev0, _, _ = format_data(dev_df[dev_df['label'] == 0], max_features, cv, tf)
    X_dev1, Y_dev1, _, _ = format_data(dev_df[dev_df['label'] == 1], max_features, cv, tf)
    X_dev2, Y_dev2, _, _ = format_data(dev_df[dev_df['label'] == args.danger_label], max_features, cv, tf)

    svc = svm.LinearSVC().fit(X_train, Y_train)

    print(len(dev_df[dev_df['label'] == 0]) / len(test_df))
    print(len(dev_df[dev_df['label'] == 1]) / len(test_df))
    print(len(dev_df[dev_df['label'] == args.danger_label]) / len(dev_df))

    make_and_print_results(svc, X_dev, Y_dev)
    make_and_print_results(svc, X_dev0, Y_dev0)
    make_and_print_results(svc, X_dev1, Y_dev1)
    make_and_print_results(svc, X_dev2, Y_dev2)


if __name__ == '__main__':
    main()
