import json
from nltk.tokenize import word_tokenize
import os.path as path
import re
import time
from nltk.stem.snowball import SnowballStemmer
from bm25_md import BM25_Model


filename_ls = ['data/QA_train.json']
dataset = []
t0 = time.time()
parent_path = path.abspath(__file__)
# path.join(pp, filename)
parent_path = parent_path.split('/')
parent_path.pop()
parent_path.pop()
parent_path = '/'.join(parent_path)
print parent_path
# exit()
stemmer = SnowballStemmer("english")


def my_tokenize(sentence):
    """ This the is tokenize function, part of the feature engineering """
    sentence = sentence.lower()
    ll = word_tokenize(sentence)
    lls = [stemmer.stem(ii) for ii in ll if re.search(r'[a-z0-9]+', ii)]

    return lls


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
        if yy in pp:
            drow['correctness'] = 1
            ac_i += 1.0

        rowss.append(drow)

    accuracy = ac_i / len(y)
    print 'accuracy: ', ac_i, '/', len(y), ': ', ac_i / len(y)
    return rowss, accuracy


for file_path in filename_ls:
    file_strs = open(path.join(parent_path, file_path)).readline()
    dataset += json.loads(file_strs)

for col in dataset:
    document_collection = col['sentences']
    bm25_query_model = BM25_Model(document_collection)
    col['model'] = bm25_query_model
pass

import csv

csvv = open('bm25_test1.csv', mode='w', )
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
        # t['prediction_sentence'] = col['sentences'][t['prediction_i']].encode('utf-8')
        # t['actual_y_sentence'] = col['sentences'][t['actual_yi']].encode('utf-8')
        writer.writerow(t)

    ddi += 1


xxx = 0

print 'EXEC:', time.time() - t0
