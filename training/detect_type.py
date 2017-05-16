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


train_set = 'training/data_set/xtrain_c.json'
test_set = 'training/data_set/xdev_c.json'

def extract(dset):
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
        if atg.count('PERSON') / l > 0.49:
            tag = 'PERSON'
        elif atg.count('LOCATION') / l > 0.49:
            tag = 'LOCATION'
        elif atg.count('ORGANIZATION') / l > 0.49:
            tag = 'ORGANIZATION'
        elif atg.count('NUMBER') / l > 0.49:
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
    _Xy = []
    for cat in train_set_by_cat:
        _Xy.extend(map(lambda qa: (qa['q'], cat), train_set_by_cat[cat]))
    _X = map(lambda tt: tt[0], _Xy)
    _y = map(lambda tt: tt[1], _Xy)
    return _X, _y, train_set_by_cat

# train_by_cat = extract(train_set)
train_Xo, train_y, train_by_cat = extract(train_set)

test_Xo, test_y, test_by_cat = extract(test_set)

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
        elif 'whom' in x.lower() or 'who' in x.lower()[0]:
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



from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier


# build_model_and_evaluate(ClassifierTemplate(LogisticRegression()),
#                          train_Xm, train_y, test_Xm, test_y, None)
#
# build_model_and_evaluate(ClassifierTemplate(RandomForestClassifier(bootstrap=False)),
#                          train_Xm, train_y, test_Xm, test_y, None)


build_model_and_evaluate(CombinedClassifier(ml=MultinomialNB(), combined=True),
                         train_Xm, train_y, test_Xm, test_y, test_Xo)

build_model_and_evaluate(CombinedClassifier(ml=LogisticRegression(C=1.0), combined=True),
                         train_Xm, train_y, test_Xm, test_y, test_Xo)

# build_model_and_evaluate(CombinedClassifier(ml=RandomForestClassifier(), combined=True),
#                          train_Xm, train_y, test_Xm, test_y, test_Xo)

# build_model_and_evaluate(CombinedClassifier(ml=DecisionTreeClassifier(), combined=True),
#                          train_Xm, train_y, test_Xm, test_y, test_Xo)

build_model_and_evaluate(CombinedClassifier(ml=SGDClassifier(penalty='elasticnet'), combined=True),
                         train_Xm, train_y, test_Xm, test_y, test_Xo)

# build_model_and_evaluate(CombinedClassifier(ml=Perceptron(), combined=True),
#                          train_Xm, train_y, test_Xm, test_y, test_Xo)
# build_model_and_evaluate(ClassifierTemplate(MLPClassifier(solver='lbfgs', alpha=1e-5,
#                                        hidden_layer_sizes=(500,), random_state=1)),
#                          train_Xm, train_y, test_Xm, test_y, test_Xo)
#
# build_model_and_evaluate(CombinedClassifier(ml=MLPClassifier(solver='lbfgs', alpha=1e-5,
#                                                              hidden_layer_sizes=(500,), random_state=1),
#                                             combined=True), train_Xm, train_y, test_Xm, test_y, test_Xo)

mmmm = 0