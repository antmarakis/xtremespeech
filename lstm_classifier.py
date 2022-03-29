"""
python lstm_classifier.py --data data-export-2021-10-04.csv --set_name1 kenya --set_type Country
python lstm_classifier.py --data data-export-2021-10-04.csv --set_name1 ek --set_type User

Basic Keras implementation of LSTMs.
"""

import argparse
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Bidirectional, GRU, Embedding, Dropout, LSTM
from keras.layers import concatenate, SpatialDropout1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Model
def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='micro')


parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
parser.add_argument('--set_name1', required=True)
parser.add_argument('--set_name2', default='default')
parser.add_argument('--set_type', required=True)
parser.add_argument('--danger_label', default=2, type=int)
parser.add_argument('--max_features', default=10000, type=int)
parser.add_argument('--max_len', default=100, type=int)
parser.add_argument('--epochs', default=5, type=int)
args = parser.parse_args()

TASK = 'extreme' if args.danger_label == 2 else 'acceptable'
DATA_PATH = 'data_splits'
max_features, max_len = args.max_features, args.max_len
f = open('res_lstm_{}_{}.txt'.format(args.set_name1, TASK), 'w')
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


def format_data(data, max_features, maxlen, tokenizer=None):
    data = data.sample(frac=1).reset_index(drop=True)
    data['text'] = data['text'].apply(lambda x: str(x))

    Y = data['label'].values # 0: Real; 1: Fake
    X = data['text']

    if not tokenizer:
        filters = "\"#$%&()*+./<=>@[\\]^_`{|}~\t\n"
        tokenizer = Tokenizer(num_words=max_features, filters=filters)
        tokenizer.fit_on_texts(list(X))

    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=maxlen)

    return X, Y, tokenizer


def read_df(train_df, dev_df, test_df):
    X_train, Y_train, tokenizer = format_data(train_df, max_features, max_len)
    X_test, Y_test, _ = format_data(test_df, max_features, max_len, tokenizer)
    X_test0, Y_test0, _ = format_data(test_df[test_df['label'] == 0], max_features, max_len, tokenizer)
    X_test1, Y_test1, _ = format_data(test_df[test_df['label'] == 1], max_features, max_len, tokenizer)
    X_test2, Y_test2, _ = format_data(test_df[test_df['label'] == args.danger_label], max_features, max_len, tokenizer)
    return X_train, Y_train, X_test, Y_test, X_test0, Y_test0, X_test1, Y_test1, X_test2, Y_test2


def build_and_train_model(X_train, Y_train):
    # Input shape
    inp = Input(shape=(max_len,))

    # Embedding and LSTM
    x = Embedding(max_features, 100)(inp)
    x = SpatialDropout1D(0.33)(x)
    x = Bidirectional(LSTM(35, return_sequences=True))(x)

    # Pooling
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])

    # Output layer
    output = Dense(1, activation='sigmoid')(conc)

    model = Model(inputs=inp, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=args.epochs, batch_size=32, verbose=1)
    return model


def convert_to_preds(results):
    """Converts probabilistic results in [0, 1] to
    binary values, 0 and 1."""
    return [1 if r > 0.5 else 0 for r in results]


def eval_model(model, X_test, Y_test):
    results = model.predict(X_test, batch_size=1, verbose=1)
    preds = convert_to_preds(results)
    acc, f1 = accuracy_score(preds, Y_test), f1_multiclass(preds, Y_test)
    print(acc)
    print(f1)
    f = open('res_lstm_{}_{}.txt'.format(args.set_name1, TASK), 'a+')
    f.write('{}; {}'.format(acc, f1))
    f.write('\n')
    f.close()


def main():
    train_df, dev_df, test_df = read_data()
    dfs = read_df(train_df, dev_df, test_df)
    # for brevity, we only evaluate on the test set
    X_train, Y_train, X_test, Y_test, X_test0, Y_test0, X_test1, Y_test1, X_test2, Y_test2 = dfs
    print(len(test_df[test_df['label'] == 0]) / len(test_df))
    print(len(test_df[test_df['label'] == 1]) / len(test_df))
    print(len(test_df[test_df['label'] == args.danger_label]) / len(test_df))
    model = build_and_train_model(X_train, Y_train)
    eval_model(model, X_test, Y_test)
    eval_model(model, X_test0, Y_test0)
    eval_model(model, X_test1, Y_test1)
    eval_model(model, X_test2, Y_test2)


if __name__ == '__main__':
    main()
