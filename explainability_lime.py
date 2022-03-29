"""
python explainability_lime.py brazil
python explainability_lime.py germany
python explainability_lime.py india
python explainability_lime.py kenya

Extracts top-contributing words with LIME
and stores them in a pickled dictionary
alongside their individual contributions.
"""

import sys, pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from simpletransformers.classification import ClassificationModel
from lime.lime_text import LimeTextExplainer

country = sys.argv[1]
class_names = ["der", "exc", "dan"]
danger_label = 2

print(country)

explainer = LimeTextExplainer(class_names=class_names)

def predict_proba(strings):
    predictions, raw_outputs = model.predict(strings)
    probabilities = np.array(raw_outputs)
    return probabilities


model = ClassificationModel("bert", "outputs_{}".format(country))

def clean_data(df):
    df = df[pd.notnull(df['label'])]
    df = df[pd.notnull(df['Text'])].reset_index()
    df['label'] = df['label'].replace({2: danger_label}).astype(int)
    df['text'] = df['text'].astype(str)
    df = df[['text', 'label']]
    return df


test_df = clean_data(pd.read_csv('data_splits/{}_test.csv'.format(country), engine='python'))

print(len(test_df))
limit = 500 # adjust this number to take into account more examples. WARNING: this takes time
if len(test_df) < limit: limit = len(test_df)
print(limit)
words = defaultdict(list)
for i in range(limit):
    explanation_test_string = "'" + test_df.iloc[i].text + "'"
    exp = explainer.explain_instance(explanation_test_string, predict_proba, num_features=3, labels=[0, 1, 2])
    for w, v in exp.as_list():
        words[w].append(abs(v)) # taking into account the absolute contribution


to_save = dict(words)
with open('explanations_lime_{}.pickle'.format(country), 'wb') as handle:
    pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

# as an example, we are sorting the dictionary by the sum of contributions
# average can be used as well
sums = {w: np.sum(values) for w, values in dict(words).items()}
sums = dict(sorted(sums.items(), key=lambda item: item[1], reverse=True))

s = '\n'.join(['{}: {}'.format(k, v) for k, v in sums.items()])
f = open('explanations_lime_{}_sum.txt'.format(country), 'w')
f.write(s)
f.close()
