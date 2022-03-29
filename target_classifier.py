"""
Example Usage (similar to base_classifier.py):
(for kenya) python target_classifier.py --set_name1 kenya
(annotator) python target_classifier.py --set_name1 ek

* set_name1 - Defines training set. Options:
    1. countries: brazil, germany, india, kenya
    2. annotators: ab, ad, an, ans, cm, cr, ec, ek, go, md, mw, ni, tl
    combinations are also accepted as comma separated lists. eg., brazil,germany
* set_name2 - Defines dev/test sets. Options are the same as set_name1. If none is given, set_name1 is used (recommended).
* set_type - Country or User
"""
import argparse, data_split
import pandas as pd
import numpy as np
from simpletransformers.classification import MultiLabelClassificationModel
from sklearn.metrics import f1_score, accuracy_score
def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='micro')


parser = argparse.ArgumentParser()
parser.add_argument('--set_name1', required=True)
parser.add_argument('--set_name2', default='default')
parser.add_argument('--epochs', default=3, type=int)
parser.add_argument('--model_type', default='bert')
parser.add_argument('--model_name', default='bert-base-multilingual-cased')
args = parser.parse_args()

DATA_PATH, TASK = 'data_splits', 'target'

idx = {
    'ethnic minorities' : 0,
    'immigrants': 1,
    'religious minorities': 2,
    'sexual minorities': 3,
    'racialized groups': 4,
    'historically oppressed caste groups': 5,
    'indigenous groups': 6,
    'women': 7,
    'any other': 8,
    'large ethnic groups': 9
}

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

def one_hot_single(i):
    """Convert single sample to one-hot"""
    one_hot = [0] * len(set(idx.values()))
    one_hot[i] = 1
    return one_hot

def to_one_hot(targets):
    """Convert all samples to one hot"""
    one_hots = []
    for target in targets:
        if not pd.notnull(target):
            one_hot = [0] * len(set(idx.values()))
            one_hots.append(one_hot)
            continue
        ts = [t.strip() for t in target.split(',')]
        one_hot = [0] * len(set(idx.values()))
        for t in ts:
            one_hot[idx[t]] = 1
        one_hots.append(one_hot)
    return one_hots

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

    train_one_hot = to_one_hot(train_df['Target (protected)'])
    dev_one_hot = to_one_hot(dev_df['Target (protected)'])
    test_one_hot = to_one_hot(test_df['Target (protected)'])

    train_df = pd.DataFrame({'text': train_df['text'], 'labels': train_one_hot}).dropna()
    dev_df = pd.DataFrame({'text': dev_df['text'], 'labels': dev_one_hot}).dropna()
    test_df = pd.DataFrame({'text': test_df['text'], 'labels': test_one_hot}).dropna()

    return train_df, dev_df, test_df


def train_model(df):
    model_args = {'manual_seed': 1, 'overwrite_output_dir': True, 'fp16': False, 'num_train_epochs': args.epochs, 'output_dir': 'outputs_target_{}'.format(args.set_name1)}
    model = MultiLabelClassificationModel(args.model_type, args.model_name, num_labels=len(set(idx.values())), args=model_args)
    model.train_model(df)
    return model


def make_predictions(model, df):
    result, model_outputs, wrong_predictions = model.eval_model(df)
    return result


def print_result(result, data_type):
    print(result)
    f = open('res_{}_{}_{}_{}.txt'.format(args.set_name1, TASK, args.model_name.replace('/', '_'), data_type), 'w')
    f.write(str(result['LRAP']))
    f.write('\n')
    f.close()


def main():
    train_df, dev_df, test_df = read_data()
    model = train_model(train_df)
    result = make_predictions(model, dev_df, 'dev')
    result = make_predictions(model, test_df, 'test')
    print_result(result)


if __name__ == '__main__':
    main()
