import json
from sklearn import metrics
filename_ls = ['data/QA_train.json']
dataset = []
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import defaultdict
from numpy import multiply
from math import sqrt
from nltk.tokenize import word_tokenize
# from nltk.corpus import Wo
import os.path as path
from collections import OrderedDict
import re

pp = path.abspath(__file__)
# path.join(pp, filename)
print pp
# exit()

def my_tokenize(sentence):
    ll = word_tokenize(sentence)
    ll = [ii.lower() for ii in ll if re.search(r'[a-z]+', ii.lower())]
    return ll


class MostRelevantSentence(object):
    def __init__(self, vectorizer, collection_matrix):
        self.vectorizer = vectorizer
        self.collection_matrix = collection_matrix
        feature_array = vectorizer.get_feature_names()
        self.features = dict()
        for fi in range(len(feature_array)):
            self.features[feature_array[fi]] = fi

    def predict(self, qX):
        ppred = []
        for x in qX:
            ppred.append(self.find_best_match2(x)[0])
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

    def find_best_match2(self, query_sent):
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

        try:
            assert ss[0][1] > ss[1][1]
        except:
            print ss,
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


def build_model_and_evaluate(clf, X, y, report=True):

    pred = clf.predict(X)
    # score
    print 'pred\ty'
    ac_i = 0
    for pp, yy in zip(pred, y):
        print pp, '\t', yy
        if pp == yy:
            ac_i += 1.0

    print('accuracy: ', ac_i / len(y))


for ff in filename_ls:
    file_strs = open(ff).readline()
    dataset += json.loads(file_strs)


for col in dataset:
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, stop_words='english', tokenizer=my_tokenize)
    document_collections = col['sentences']
    # col['tf-idf'] = tfidf_vectorizer.fit_transform(document_collections)
    # col['vectorizer'] = tfidf_vectorizer
    # col['vocabulary'] = tfidf_vectorizer.get_feature_names()
    col['model'] = MostRelevantSentence(vectorizer=tfidf_vectorizer,
                                        collection_matrix=tfidf_vectorizer.fit_transform(document_collections))
pass


for col in dataset:
    qX = [i['question'] for i in col['qa']]
    qy = [i['answer_sentence'] for i in col['qa']]
    model = col['model']
    build_model_and_evaluate(model, qX[:10], qy[:10])
    exit()

xxx = 0


