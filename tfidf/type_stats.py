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

import nltk



for ff in filename_ls:
    file_strs = open(path.join(pp, ff)).readline()
    dataset += json.loads(file_strs)

for col in dataset:
    for qa in col['qa']:
        q = qa['question']
        a = qa['answer_sentence']
        qt = nltk.pos_tag(word_tokenize(q))

        # if




