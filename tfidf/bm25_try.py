from gensim.summarization.bm25 import BM25, PARAM_K1,PARAM_B, EPSILON
PARAM_K1 = 1.9
# from gensim.summarization.bm25 import BM25, PARAM_K1
print PARAM_K1
exit()
import json
from sklearn import metrics
filename_ls = ['data/QA_train.json']
dataset = []
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import defaultdict

from nltk.tokenize import word_tokenize
import os.path as path
from collections import OrderedDict
import re

import time

from nltk.stem.snowball import SnowballStemmer

t0 = time.time()
path_prefix = path.abspath(__file__)
# path.join(pp, filename)
path_prefix = path_prefix.split('/')
path_prefix.pop()
path_prefix.pop()
path_prefix = '/'.join(path_prefix)
print path_prefix


# class BM25MD(BM25):
#     def

class BestMatchBM2PredictionModel():

    def __init__(self, doc_collection):
        self.doc_collection = doc_collection
        self.bm

    def predict(self):
        pass

for ff in filename_ls:
    file_strs = open(path.join(path_prefix, ff)).readline()
    dataset += json.loads(file_strs)



for col in dataset:
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=1, use_idf=True,
                                       stop_words='english', tokenizer=my_tokenize, norm='l2', sublinear_tf=True)

    document_collections = col['sentences']

    tfidf_matrix = tfidf_vectorizer.fit_transform(document_collections)
    col['model'] = MostRelevantSentenceModel(vectorizer=tfidf_vectorizer,
                                             collection_matrix=tfidf_matrix)
pass
