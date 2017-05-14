#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import math
from six import iteritems
from nltk import FreqDist, word_tokenize
from collections import defaultdict

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
import re

def my_tokenize(sentence):
    """ This the is tokenize function, part of the feature engineering """
    sentence = sentence.lower()
    ll = word_tokenize(sentence)
    lls = [stemmer.stem(ii) for ii in ll if re.search(r'[a-z0-9]+', ii)]
    return lls

def dict_add(d1, d2):
    cc = defaultdict(float)
    for key in set(d1.keys()+d2.keys()):
        cc[key] += d1[key] + d2[key]

    return cc

class BM25_Model(object):

    def __init__(self, document_collection, K1=0.3, B=0.01, K3=1.0, EPS=0.25, tokenizer=None):
        if not tokenizer:
            self.tokenizer = my_tokenize
        else:
            self.tokenizer = tokenizer
        self.document_collection_length = len(document_collection)
        self.avg_doc_length = sum(map(lambda x: float(len(x)), document_collection)) / self.document_collection_length
        self.document_collection = [self.tokenizer(doc) for doc in document_collection]
        self.f = []
        self.df = defaultdict(int)
        self.bm25_idf = defaultdict(float)
        self.idf_1 = defaultdict(float)
        self.average_idf = -1
        self.K1 = K1
        self.K3 = K3
        self.EPSILON = EPS
        self.B = B
        self.inverted_index = defaultdict(list)
        self.initialize()

    def initialize(self):
        for index, document in enumerate(self.document_collection):
            frequencies = FreqDist(document)
            self.f.append(frequencies)

            for word, freq in iteritems(frequencies):
                self.df[word] += 1
                self.inverted_index[word].append(index)

        for word, freq in self.df.items():
            self.bm25_idf[word] = math.log(self.document_collection_length - freq + 0.5) - math.log(freq + 0.5)
            # self.idf_1 = math.log((self.document_collection_length - freq))
        self.average_idf = sum(map(lambda k: float(self.bm25_idf[k]), self.bm25_idf.keys())) / len(self.bm25_idf.keys())

    def predict(self, queryX, limit=3):
        q_prediction = []
        for query in queryX:
            ls = self.bm25_get_most_relevant(query)[:limit]
            q_prediction.append([a for a, b in ls])
        return q_prediction

    def bm25_get_most_relevant(self, query):
        query_tks = self.tokenizer(query)
        scores = defaultdict(float)
        for q_token in query_tks:
            for doc_index in self.inverted_index[q_token]:
                idf = self.bm25_idf[q_token] if self.bm25_idf[q_token] >= 0 else self.EPSILON * self.average_idf
                top = self.f[doc_index][q_token] * (self.K1 + 1)
                below = self.f[doc_index][q_token] + self.K1 * (
                1 - self.B + self.B * self.document_collection_length / self.avg_doc_length)
                scores[doc_index] += idf * top / below
        prels = scores.items()
        sorted_socres = sorted(prels, key=lambda (k, v): v, reverse=True)
        return sorted_socres



