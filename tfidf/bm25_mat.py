#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import math
from six import iteritems
from six.moves import xrange
from nltk import FreqDist, word_tokenize

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
import re



def my_tokenize(sentence):
    """ This the is tokenize function, part of the feature engineering """
    sentence = sentence.lower()
    ll = word_tokenize(sentence)
    lls = [stemmer.stem(ii) for ii in ll if re.search(r'[a-z0-9]+', ii)]
    return lls


class BM25_Model(object):

    def __init__(self, document_collection, K1, B, K3, EPS, tokenizer=None):
        if not tokenizer:
            self.tokenizer = my_tokenize
        else:
            self.tokenizer = tokenizer
        self.document_collection_length = len(document_collection)
        self.avgdl = sum(map(lambda x: float(len(x)), document_collection)) / self.document_collection_length
        self.document_collection = [self.tokenizer(doc) for doc in document_collection]
        self.f = []
        self.df = {}
        self.idf = {}
        self.average_idf = -1
        self.K1 = K1
        self.K3 = K3
        self.EPSILON = EPS
        self.B = B

        self.initialize()

    def initialize(self):
        for document in self.document_collection:
            # frequencies = {}
            # for word in document:
            #     if word not in frequencies:
            #         frequencies[word] = 0
            #     frequencies[word] += 1
            frequencies = FreqDist(document)
            self.f.append(frequencies)

            for word, freq in iteritems(frequencies):
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1

        for word, freq in iteritems(self.df):
            self.idf[word] = math.log(self.document_collection_length - freq + 0.5) - math.log(freq + 0.5)

        self.average_idf = sum(map(lambda k: float(self.idf[k]), self.idf.keys())) / len(self.idf.keys())

    def _scoring_(self, query, doc_index):
        score = 0
        for word in query:
            if word not in self.f[doc_index]:
                continue
            idf = self.idf[word] if self.idf[word] >= 0 else self.EPSILON * self.average_idf
            score += (idf * self.f[doc_index][word] * (self.K1 + 1)
                      / (self.f[doc_index][word] + self.K1 * (1 - self.B + self.B * self.document_collection_length / self.avgdl)))
        return score

    def get_most_relevant(self, query):
        query_tks = self.tokenizer(query)
        scores = []
        for index in range(self.document_collection_length):
            score = self._scoring_(query_tks, index)
            scores.append(score)

        sorted_socres = sorted(list(enumerate(scores)), key=lambda (k, v): v)
        return sorted_socres



