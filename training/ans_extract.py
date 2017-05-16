import json
from nltk.tokenize import word_tokenize
import os.path as path
import time
from collections import defaultdict
from nltk.corpus import stopwords
stop_english = set(stopwords.words('english'))
from nltk.stem import SnowballStemmer
import re
import sys
from sklearn import metrics
import random

stemmer = SnowballStemmer(language='english')

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

def extract_x(qa):
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

    q = dict()
    q['q'] = qtk
    q['a'] = qa['answer']
    q['as'] = qa['ans_sent']
    q['type'] = tag

    return q

extract_Xy = map(extract_set, all_qa)


def rd(a, b):
    d = defaultdict(list)
    ts = set(a.keys() + b.keys())
    for t in ts:
        d[t].extend(a[t])
        d[t].extend(b[t])
    return d

train_set_by_cat = reduce(rd, extract_Xy)


qass = map(extract_x, all_qa)

vvv = 0

from nltk import pos_tag, RegexpParser, Tree

grammar = """

    NE:
    {<DT><CD><J.*>+<N.*>+}
    {<DT>?<CD><J.*>+<N.*>+}
    {<DT><CD>?<J.*>+<N.*>+}
    {<DT>?<CD>?<J.*>?<N.*>+}
    # {<DT>?<N.*>+<IN>?<N.*>+}
    <``>{<.*>+}<''>
    <BRA>{<.*>+}<BRB>

"""

rg_parser = RegexpParser(grammar=grammar)

def guess(qa):
    qd = qa['q']
    ast = qa['as']

    def cg(tg):
        if tg[1] == '(':
            return tg[0], 'BRA'
        if tg[1] == ')':
            return tg[0], 'BRB'
        return tg
    qtks = map(cg, pos_tag(word_tokenize(qd)))
    atks = map(cg, pos_tag(word_tokenize(ast)))
    # qtkcs = map(lambda x: x + (stemmer.stem(x[0]),), qtks)
    # atkcs = map(lambda x: x + (stemmer.stem(x[0]),), atks)
    # qstem = map(lambda x: x[2], qtkcs)

    def ft(qtk):
        if re.search(r'W.*', qtk[1]):
            return '?', '?'
        if re.search(r'V.*', qtk[1]) and qtk[1] != 'VBG':
            return '|', '|'
        if qtk[1] in ['MD', 'IN', 'RB', ',', '.', '(', ')']:
            return '|', '|'
        return qtk[0], qtk[1]
    # qa['qtk'] = map(ft, qtkcs)

    def ety(tn):
        if isinstance(tn, Tree):
            return tn.flatten()

    qa['a_ety'] = rg_parser.parse(atks)
    qa['_aety'] = map(ety, rg_parser.parse(atks))
    qa['q_ety'] = rg_parser.parse(qtks)
    qa['_qety'] = map(ety, rg_parser.parse(qtks))



    # core = defaultdict(list)
    # pson = 0
    # for qt in qa['qtk']:
    #     if pson == 0 and qt[1] == '|':
    #         pass
    #     if pson != -1 and qt[1] == '?':
    #         pson = 1
    #     elif pson == 1 and qt[1] != '?' and qt[1] != '|':
    #         core['core'].append(qt)
    #     elif qt[1] != '?' and qt[1] != '|':
    #         core['rest'].append(qt)
    #     elif qt[1] == '|':
    #         pson = -1
    # qa['core'] = core

    def rm(atkc):
        if atkc[2] in qstem or atkc[2] in stop_english:
            if atkc[1] not in ['DT']:
                return '*', '*'
            else:
                return atkc[0], atkc[1]
        if (re.search(r'V.*', atkc[1]) and atkc[1] != 'VBG'):
            return '|', '|'
        if atkc[1] in ['MD', 'IN', 'RB', ',', '.', '(', ')']:
            return '|', '|'
        if re.search(r'W.*', atkc[1]):
            return '|', '|'
        if re.search(r'N.*', atkc[1]):
            return atkc[0], 'NE'
        return atkc[0], atkc[1]
    # qa['atk'] = map(rm, atkcs)


    cccc=0
    return qa


pre = map(guess, qass[1400:1600])


cccc = 0




