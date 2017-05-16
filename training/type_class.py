import json
from nltk.tokenize import word_tokenize
import os.path as path
import time
from collections import defaultdict
from nltk.corpus import stopwords
stop_english = set(stopwords.words('english'))
import sys
from sklearn import metrics
import random

cur_path = path.dirname(__file__)
parent_path = path.dirname(cur_path)
print parent_path


dset = 'training/data_set/xtrain_ner.json'

dataset = json.loads(open(path.join(parent_path, dset)).readline())
all_qa = []

for col in dataset:
    all_qa.extend(col['qa'])


#
# def ext1(tk):
#     return map(lambda x: x[0].lower(), tk)

def extract_set(qa):
    qtk, atg = qa['question'], qa['answer']
    qtk = map(lambda x: x[0].lower(), qtk)
    atg = map(lambda x: x[1], atg)
    tag = 'O'
    if atg.count('PERSON') == len(atg):
        tag = 'PERSON'
    elif atg.count('LOCATION') == len(atg):
        tag = 'LOCATION'
    elif atg.count('ORGANIZATION') == len(atg):
        tag = 'ORGANIZATION'
    elif atg.count('NUMBER') == len(atg):
        tag = 'NUMBER'
    d = defaultdict(list)
    d[tag].append(qtk)

    return d


extract_Xy = map(extract_set, all_qa)


def rd(a, b):
    d = defaultdict(list)
    ts = set(a.keys() + b.keys())
    for t in ts:
        d[t].extend(a[t])
        d[t].extend(b[t])
    return d

train_set_by_cat = reduce(rd, extract_Xy)

def sli(lt):
    i, l = lt
    lg = len(l)
    random.shuffle(l)
    cut = int(0.9*lg)
    tr = l[:cut]
    ts = l[cut:]
    return i, tr, ts

train_test = map(sli, train_set_by_cat.items())

train_Xy = []
test_Xy = []
for group in train_test:
    tag = group[0]
    train_Xy.extend(map(lambda g: (g, tag), group[1]))
    test_Xy.extend(map(lambda g: (g, tag), group[2]))


# tr_Xy = reduce(lambda )
random.shuffle(train_Xy)

train_X = map(lambda t: t[0], train_Xy)
train_y = map(lambda t: t[1], train_Xy)

test_X = map(lambda t: t[0], test_Xy)
test_y = map(lambda t: t[1], test_Xy)

from sklearn.feature_extraction import DictVectorizer
from nltk import FreqDist
vc = DictVectorizer()

train_X = vc.fit_transform(map(lambda tks: FreqDist(tks), train_X))
test_X = vc.transform(map(lambda tks: FreqDist(tks), test_X))
ccc = 0


def build_model_and_evaluate(clf, train_X, train_y, test_X, test_y, report=True):
    # training
    clf.fit(train_X, train_y)
    # test
    pred = clf.predict(test_X)
    # score
    accuracy = metrics.accuracy_score(test_y, pred)
    if report:
        print('-' * 100)
        print('classifier:')
        print(clf)

        print("macro f1 score:   %0.3f" % metrics.f1_score(test_y, pred, average='macro'))
        print"accuracy:   %0.3f" % accuracy, '\n\n'
        print(metrics.classification_report(test_y, pred))
        print()
        print(metrics.confusion_matrix)
    print()
    return accuracy



from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier




# build_model_and_evaluate(MultinomialNB(), train_X, train_y, test_X, test_y)
build_model_and_evaluate(LogisticRegression(), train_X, train_y, test_X, test_y)
build_model_and_evaluate(RandomForestClassifier(), train_X, train_y, test_X, test_y)
