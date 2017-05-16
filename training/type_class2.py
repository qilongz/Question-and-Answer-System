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


dset = 'training/data_set/xdev_c.json'
dataset = []
texts = open(path.join(parent_path, dset)).readlines()
for t in texts:
    dataset.append(json.loads(t))

all_qa = []

for col in dataset:
    all_qa.extend(col['qa'])


#
# def ext1(tk):
#     return map(lambda x: x[0].lower(), tk)

def extract_set(qa):
    qtk, atg = qa['question'], qa['answer']
    # qtk = map(lambda x: x[0].lower(), qtk)
    atg = map(lambda x: x[1], atg)
    tag = 'O'
    l = float(len(atg))
    if atg.count('PERSON')/l > 0.49:
        tag = 'PERSON'
    elif atg.count('LOCATION')/l > 0.49:
        tag = 'LOCATION'
    elif atg.count('ORGANIZATION')/l > 0.49:
        tag = 'ORGANIZATION'
    elif atg.count('NUMBER')/l > 0.49:
        tag = 'NUMBER'
    d = defaultdict(list)
    q = dict()
    q['q'] = qtk
    q['a'] = qa['answer']
    q['as'] = qa['ans_sent']
    d[tag].append(q)

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

# train_set_by_cat['O'].extend(train_set_by_cat['NUMBER'])
# del train_set_by_cat['NUMBER']
# train_set_by_cat['O'].extend(train_set_by_cat['ORGANIZATION'])
# del train_set_by_cat['ORGANIZATION']
# train_set_by_cat['O'].extend(train_set_by_cat['PERSON'])
# del train_set_by_cat['PERSON']


def sli(lt):
    i, l = lt
    lg = len(l)
    random.shuffle(l)
    cut = int(0.9*lg)
    tr = l[:cut]
    ts = l[cut:]
    return i, tr, ts

train_test = map(sli, train_set_by_cat.items())

train_Xyo = []
test_Xy = []
for group in train_test:
    tag = group[0]
    train_Xyo.extend(map(lambda g: (g, tag), group[1]))
    test_Xy.extend(map(lambda g: (g, tag), group[2]))


# tr_Xy = reduce(lambda )
# random.shuffle(train_Xy)

train_Xo = map(lambda t: t[0]['q'], train_Xyo)
train_y = map(lambda t: t[1], train_Xyo)

test_Xo = map(lambda t: t[0]['q'], test_Xy)
test_y = map(lambda t: t[1], test_Xy)

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
import re

def my_tokenize(sentence):
    """ This the is tokenize function, part of the feature engineering """
    sentence = sentence.lower()
    ll = word_tokenize(sentence)
    lls = [stemmer.stem(ii) for ii in ll if re.search(r'[a-z0-9]+', ii)]

    return lls


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english', tokenizer=my_tokenize)

train_Xm = cv.fit_transform(train_Xo)

test_Xm = cv.transform(test_Xo)

from sklearn.decomposition import TruncatedSVD

# svd = TruncatedSVD(n_components=5000)
#
# train_X = svd.fit_transform(train_Xm)
#
# test_X = svd.transform(test_Xm)


ppppp=0


def build_model_and_evaluate(clf, train_Xmat, train_y, test_Xmat, test_y, test_Xo, report=True):
    # training
    clf.fit(train_Xmat, train_y)
    # test
    pred = clf.predict(Xmat=test_Xmat, Xo=test_Xo)
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
from sklearn.neural_network import MLPClassifier


class ClassifierTemplate:
    def __init__(self, ml):
        self.ml = ml
        return

    def fit(self, tX, ty):
        self.ml.fit(tX, ty)
        return

    def predict(self, Xmat=None, Xo=None):
        return self.ml.predict(Xmat)

    def __str__(self):
        return str(self.ml)


from nltk import word_tokenize
class CombinedClassifier(ClassifierTemplate, object):
    def __init__(self, ml, combined=False):
        ClassifierTemplate.__init__(self, ml)
        self.combined = combined
        return

    def fit(self, tX, ty):
        self.ml.fit(tX, ty)
        return

    def _pred_(self, x, xm=None):
        if 'where' in x.lower():
            return 'LOCATION'
        elif 'when' in x.lower():
            return 'NUMBER'
        elif 'what rate' in x.lower():
            return 'NUMBER'
        elif 'whom' in x.lower() or 'who' in x.lower():
            return 'PERSON'
        elif 'which' in x.lower():
            if 'team' in x.lower():
                return 'ORGANIZATION'
            else:
                return 'O'
        else:
            return 'O'

    def _pred_with_ml_(self, x, xm=None):
        if 'where' in x.lower():
            return 'LOCATION'
        elif 'when' in x.lower():
            return 'NUMBER'
        elif 'how much' in x.lower():
            return 'NUMBER'
        elif 'how many' in x.lower():
            return 'NUMBER'
        elif 'what rate' in x.lower():
            return 'NUMBER'
        elif 'whom' in x.lower() or 'who' in x.lower():
            return 'PERSON'
        else:
            return 'O'

    def predict(self, Xmat=None, Xo=None):
        assert (Xo or Xmat is not None) and self.ml
        if Xo and Xmat is None:
            pl = []
            for x in Xo:
                pl.append(self._pred_(x))
            return pl
        elif self.ml and Xmat is not None and not Xo:
            return self.ml.predict(Xo)
        elif self.ml and Xmat is not None and Xo:
            rule_prediction = []
            for x in Xo:
                rule_prediction.append(self._pred_(x))
            ml_prediction = self.ml.predict(Xmat)

            def overwrite(rl_p, ml_p):
                if rl_p != ml_p:
                    if rl_p == 'O':
                        return ml_p
                    elif ml_p == 'ORGANIZATION':
                        return ml_p
                    else:
                        return rl_p
                return ml_p

            return map(overwrite, rule_prediction, ml_prediction)




    def __str__(self):
        if self.combined and self.ml:
            return 'Combined ########\n' + str(self.ml)
        else:
            return 'WTF!'

pass

# build_model_and_evaluate(MultinomialNB(), train_X, train_y, test_X, test_y)
build_model_and_evaluate(ClassifierTemplate(LogisticRegression()),
                         train_Xm, train_y, test_Xm, test_y, None)

build_model_and_evaluate(ClassifierTemplate(RandomForestClassifier()),
                         train_Xm, train_y, test_Xm, test_y, None)

build_model_and_evaluate(CombinedClassifier(ml=None),
                         train_Xm, train_y, test_Xm, test_y, None)

build_model_and_evaluate(CombinedClassifier(ml=LogisticRegression(), combined=True),
                         train_Xm, train_y, test_Xm, test_y, test_Xo)

# build_model_and_evaluate(CombinedClassifier(ml=LogisticRegression(), combined=True),
#                          train_Xm, train_y, test_Xm, test_y, test_Xo)



# build_model_and_evaluate(ClassifierTemplate(MLPClassifier(solver='lbfgs', alpha=1e-5,
#                                        hidden_layer_sizes=(500,), random_state=1)),
#                          train_Xm, train_y, test_Xm, test_y, test_Xo)
#
# build_model_and_evaluate(CombinedClassifier(ml=MLPClassifier(solver='lbfgs', alpha=1e-5,
#                                                              hidden_layer_sizes=(500,), random_state=1),
#                                             combined=True), train_Xm, train_y, test_Xm, test_y, test_Xo)
#
# mmmm = 0