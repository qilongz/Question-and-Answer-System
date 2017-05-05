import json
from sklearn import metrics
filename_ls = ['data/QA_train.json']
dataset = []
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import defaultdict
from numpy import multiply
from math import sqrt
from nltk.tokenize import word_tokenize
import os.path as path
from collections import OrderedDict
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer as WNL
import time
from sklearn.feature_extraction import DictVectorizer
from nltk import FreqDist, DictionaryProbDist
from operator import add
import math
import numpy



t0 = time.time()
pp = path.abspath(__file__)
# path.join(pp, filename)
pp = pp.split('/')
pp.pop()
pp.pop()
pp = '/'.join(pp)
print pp
# exit()
wnl = WNL()


def my_tokenize(sentence):
    """ This the is tokenize function, part of the feature engineering """

    sentence = sentence.lower()
    ll = word_tokenize(sentence)

    # llx = nltk.pos_tag(ll)
    # llw = []
    # for ww in llx:
    #     if re.search(r'^V', ww[1]):
    #         llw.append(wnl.lemmatize(ww[0], 'v'))
    #     elif re.search(r'^N', ww[1]):
    #         llw.append(wnl.lemmatize(ww[0], 'n'))
    #     elif re.search(r'[a-z0-9]+', ww[0]):
    #         llw.append(ww[0])

    ll = [ii for ii in ll if re.search(r'[a-z0-9]+', ii)]
    return ll


class MostRelevantSentence(object):
    def __init__(self, vectorizer, collection_matrix):
        self.vectorizer = vectorizer
        self.collection_matrix = collection_matrix
        feature_array = vectorizer.get_feature_names()
        self.features = dict()
        for fi in range(len(feature_array)):
            self.features[feature_array[fi]] = fi

        # for j in range(self.collection_matrix.shape[0]):
        #     print self.collection_matrix[j,:].sum()
        #     time.sleep(0.5)
            # if self.collection_matrix[j, :].sum() != numpy.float64(1.0):
            #     x = self.collection_matrix[j, :].sum()
            #     print type(x)
            #     raise Exception()

    def predict(self, qX):
        ppred = []
        for x in qX:
            ppred.append(self.inverted_index_query(x)[0])
        return ppred

    def find_best_match(self, query_sent):
        """
        compare question sentence with each sentence in article, 
        using cosine distance find the best match
        
        :param query_sent: 
        :return: 
        """

        score = defaultdict(float)
        vec = self.vectorizer.transform([query_sent])

        # lenn = len(self.collection_vecs)
        for ii in range(self.collection_matrix.shape[0]):
            v1 = self.collection_matrix[ii, :]
            score[ii] = self.cosine_d(v1, vec)

        ss = sorted(score.items(), key=lambda (k, v): v, reverse=True)

        try:
            assert ss[0][1] >= ss[1][1]
        except:
            print ('WTF?')
        return ss[0]

    def inverted_index_query(self, query_sent):
        """
        now we implement inverted index to handle query
        
        :param query_sent: 
        :return: 
        
        """
        query_words = my_tokenize(query_sent)

        # query_words = [for i in query_sents if re.search()]

        score = defaultdict(float)

        for w in query_words:
            try:
                col_i = self.features[w]
                inverted_ix = self.collection_matrix[:, col_i]
                for di in range(inverted_ix.shape[0]):
                    score[di] += inverted_ix[di, 0]
            except KeyError:
                pass

        ss = sorted(score.items(), key=lambda (k, v): v, reverse=True)

        # try:
        #     assert ss[0][1] >= ss[1][1]
        # except:
        #     print ss,
        if ss:
            return ss[0]
        else:
            return -1, 0

    def cosine_d(self, a, b):
        # asp = a.shape
        # bsp = b.shape
        # bb = b.transpose()
        # xx = multiply(a, b.transpose())
        dotp = multiply(a, b.transpose()).sum()
        base = sqrt(multiply(multiply(a, a.transpose()).sum(), multiply(b, b.transpose()).sum()))
        # assert base == 1.0
        # print base
        if base:
            return dotp / base
        else:
            return 0


def build_model_and_evaluate(clf, X, y, report=False):
    rowss = []
    pred = clf.predict(X)
    # score
    print 'pred\ty'
    ac_i = 0
    qi = 0
    for pp, yy in zip(pred, y):
        drow = dict()
        if report:
            print pp, '\t', yy
        drow['question_i'] = qi
        drow['prediction_i'] = pp
        drow['actual_yi'] = yy
        qi += 1
        if pp == yy:
            drow['correctness'] = 1
            ac_i += 1.0

        rowss.append(drow)

    accuracy = ac_i / len(y)
    print('accuracy: ', ac_i / len(y))
    return rowss, accuracy


for ff in filename_ls:
    file_strs = open(path.join(pp, ff)).readline()
    dataset += json.loads(file_strs)


for col in dataset:
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=1, use_idf=True,
                                       stop_words='english', tokenizer=my_tokenize, norm='l2', sublinear_tf=True)
    # tfidf_vectorizer = CountVectorizer(max_df=1.0, min_df=1,
    #                                    stop_words=None, tokenizer=my_tokenize)
    document_collections = col['sentences']

    tfidf_matrix = tfidf_vectorizer.fit_transform(document_collections)
    col['model'] = MostRelevantSentence(vectorizer=tfidf_vectorizer,
                                        collection_matrix=tfidf_matrix)
pass







import csv
csvv = open('names.csv', mode='w',)
fieldnames = ['document_i', 'question_i', 'prediction_i',
              'actual_yi', 'correctness', 'question', 'prediction_sentence', 'actual_y_sentence']
writer = csv.DictWriter(csvv, fieldnames=fieldnames, )
writer.writeheader()

ddi = 0
for col in dataset:

    qX = [i['question'] for i in col['qa']]
    qy = [i['answer_sentence'] for i in col['qa']]
    model = col['model']
    table, acc = build_model_and_evaluate(model, qX, qy)
    for t in table:
        t['document_i'] = ddi
        t['question'] = qX[t['question_i']].encode('utf-8')
        t['prediction_sentence'] = col['sentences'][t['prediction_i']].encode('utf-8')
        t['actual_y_sentence'] = col['sentences'][t['actual_yi']].encode('utf-8')
        writer.writerow(t)


    ddi += 1

    # if ddi > 10:
    #     exit()

xxx = 0


print 'EXEC:', time.time() - t0
