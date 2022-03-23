"""
Example Usage:
(bare-bones) python base_classifier.py --set_name1 kenya
(for brazil) python base_classifier.py --set_name1 brazil --set_type Country --model_name neuralmind/bert-base-portuguese-cased
(for germany) python base_classifier.py --set_name1 germany --set_type Country --model_name bert-base-german-cased
(for india) python base_classifier.py --set_name1 india --set_type Country --model_name monsoon-nlp/hindi-bert
(for kenya) python base_classifier.py --set_name1 kenya --model_name flax-community/bert-base-uncased-swahili
(all countries) python base_classifier.py --set_name1 brazil,germany,india,kenya --set_name2 brazil,germany,india,kenya
(annotator) python base_classifier.py --set_name1 ek

Data Labels:
    0 = derogatory extreme speech
    1 = exclusionary extreme speech
    2 = dangerous extreme speech

* set_name1 - Defines training set. Options:
    1. countries: brazil, germany, india, kenya
    2. annotators: ab, ad, an, ans, cm, cr, ec, ek, go, md, mw, ni, tl
    combinations are also accepted as comma separated lists. eg., brazil,germany
* set_name2 - Defines dev/test sets. Options are the same as set_name1. If none is given, set_name1 is used (recommended).
* set_type - Country or User
* danger_label - whether Dangerous Speech is labeled as 1 or 2. In the first case, we have binary classification (M vs. R).
In the second case, we have three-way classification (derogatory vs. exclusionary vs. dangerous)
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
parser.add_argument('--danger_label', default=2, type=int) # 1: acceptable, 2: extreme
parser.add_argument('--epochs', default=3, type=int)
parser.add_argument('--model_type', default='bert')
parser.add_argument('--model_name', default='bert-base-multilingual-cased')
args = parser.parse_args()

TASK = 'extreme' if args.danger_label == 2 else 'acceptable'
DATA_PATH = 'data_splits'

def read_data():
    """
    Read training, dev and test sets. If multiple sets are given (via a comma-separated list),
    read them separately and concatenate each set.
    """
    if ',' in args.set_name1:
        train_data = []
        for name in args.set_name1.split(','):
            train_data.append(pd.read_csv('{}/{}_train.csv'.format(DATA_PATH, name), engine='python'))
        train_df = pd.concat(train_data)
    else:
        train_df = pd.read_csv('{}/{}_train.csv'.format(DATA_PATH, args.set_name1), engine='python')

    if args.set_name2 == 'default':
        if ',' in args.set_name1: print('ERROR! set_name1 must not be a list')
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
    """
    Basic cleanup, plus replacing dangerous speech label with the given (or default) value
    """
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


def train_model(df):
    """
    Finetune a pretrained model
    """
    model_args = {'manual_seed': 1, 'overwrite_output_dir': True, 'fp16': False,
                  'num_train_epochs': args.epochs, 'output_dir': 'outputs_{}_{}'.format(TASK, args.set_name1)}
    model = ClassificationModel(args.model_type, args.model_name, num_labels=args.danger_label+1, args=model_args)
    model.train_model(df)
    return model


def make_predictions(model, df):
    """
    Returns metrics (eg., f1-score). For full results, you can modify this function to return (the raw) model_outputs,
    which you would then have to convert to predictions (for example, by picking the one with maximum confidence)
    """
    result, model_outputs, wrong_predictions = model.eval_model(df, f1=f1_multiclass, acc=accuracy_score)
    return result


def print_predictions(result, dev_or_test):
    print(result)
    f = open('res_{}_{}_{}_{}_{}.txt'.format(args.set_name1, args.set_name2, TASK, args.model_name.replace('/', '_'), dev_or_test), 'a+')
    f.write(str(result['f1']))
    f.write('\n')
    f.close()


def main():
    train_df, dev_df, test_df = read_data()
    model = train_model(train_df)

    dev_res = make_predictions(model, dev_df)
    dev0_res = make_predictions(model, dev_df[dev_df['label'] == 0])
    dev1_res = make_predictions(model, dev_df[dev_df['label'] == 1])
    dev2_res = make_predictions(model, dev_df[dev_df['label'] == args.danger_label])

    print_predictions(dev_res, 'dev')
    print_predictions(dev0_res, 'dev')
    print_predictions(dev1_res, 'dev')
    print_predictions(dev2_res, 'dev')

    test_res = make_predictions(model, test_df)
    test0_res = make_predictions(model, test_df[test_df['label'] == 0])
    test1_res = make_predictions(model, test_df[test_df['label'] == 1])
    test2_res = make_predictions(model, test_df[test_df['label'] == args.danger_label])

    print_predictions(test_res, 'test')
    print_predictions(test0_res, 'test')
    print_predictions(test1_res, 'test')
    print_predictions(test2_res, 'test')


if __name__ == "__main__":
    main()
