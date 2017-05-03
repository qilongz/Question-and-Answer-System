import json
from sklearn import metrics
filename_ls = ['data/QA_train.json']
dataset = []
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import defaultdict
from numpy import multiply
from math import sqrt


class MostRelevantSentence(object):
    def __init__(self, vectorizer, collection_vecs):
        self.vectorizer = vectorizer
        self.collection_vecs = collection_vecs
        self.features = vectorizer.get_feature_names()

    def predict(self, qX):
        ppred = []
        for x in qX:
            ppred.append(self.find_best_match(x)[0])
        return ppred

    def find_best_match(self, qs):
        score = defaultdict(float)
        vec = self.vectorizer.transform([qs])
        # lenn = len(self.collection_vecs)
        for ii in range(self.collection_vecs.shape[0]):
            v1 = self.collection_vecs[ii, :]
            score[ii] = self.cosine_d(v1, vec)

        ss = sorted(score.items(), key=lambda (k, v): v, reverse=True)

        try:
            assert ss[0][1] > ss[1][1]
        except:
            print ss
        return ss[0]

    def find_best_match2(self, qs):
        score = defaultdict(float)
        vec = self.vectorizer.transform([qs])
        # lenn = len(self.collection_vecs)
        for ii in range(self.collection_vecs.shape[0]):
            v1 = self.collection_vecs[ii, :]
            score[ii] = self.cosine_d(v1, vec)

        ss = sorted(score.items(), key=lambda (k, v): v, reverse=True)

        try:
            assert ss[0][1] > ss[1][1]
        except:
            print ss
        return ss[0]

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
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, stop_words='english')
    document_collections = col['sentences']
    # col['tf-idf'] = tfidf_vectorizer.fit_transform(document_collections)
    # col['vectorizer'] = tfidf_vectorizer
    # col['vocabulary'] = tfidf_vectorizer.get_feature_names()
    col['model'] = MostRelevantSentence(vectorizer=tfidf_vectorizer,
                                        collection_vecs=tfidf_vectorizer.fit_transform(document_collections))
pass


for col in dataset:
    qX = [i['question'] for i in col['qa']]
    qy = [i['answer_sentence'] for i in col['qa']]
    model = col['model']
    build_model_and_evaluate(model, qX[:10], qy[:10])
    exit()

xxx = 0


