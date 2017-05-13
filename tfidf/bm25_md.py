#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import math
from six import iteritems
from six.moves import xrange


class BM25_Model(object):

    def __init__(self, corpus, K1, B, K3, EPSILON):
        self.corpus_size = len(corpus)
        self.avgdl = sum(map(lambda x: float(len(x)), corpus)) / self.corpus_size
        self.corpus = corpus
        self.f = []
        self.df = {}
        self.idf = {}
        self.average_idf = -1
        self.K1 = K1
        self.K3 = K3
        self.EPSILON = EPSILON
        self.B = B

        self.initialize()

    def initialize(self):
        for document in self.corpus:
            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.f.append(frequencies)

            for word, freq in iteritems(frequencies):
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1

        for word, freq in iteritems(self.df):
            self.idf[word] = math.log(self.corpus_size-freq+0.5) - math.log(freq+0.5)

        self.average_idf = sum(map(lambda k: float(self.idf[k]), self.idf.keys())) / len(self.idf.keys())

    def get_score(self, document, index, average_idf):
        score = 0
        for word in document:
            if word not in self.f[index]:
                continue
            idf = self.idf[word] if self.idf[word] >= 0 else self.EPSILON * average_idf
            score += (idf * self.f[index][word] * (self.K1 + 1)
                      / (self.f[index][word] + self.K1 * (1 - self.B + self.B * self.corpus_size / self.avgdl)))
        return score

    def get_scores(self, document, average_idf):
        scores = []
        for index in range(self.corpus_size):
            score = self.get_score(document, index, average_idf)
            scores.append(score)
        return scores


def get_bm25_weights(corpus):
    bm25 = BM25(corpus)
    average_idf = sum(map(lambda k: float(bm25.idf[k]), bm25.idf.keys())) / len(bm25.idf.keys())

    weights = []
    for doc in corpus:
        scores = bm25.get_scores(doc, average_idf)
        weights.append(scores)

    return weights
