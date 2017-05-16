import json
from nltk.tokenize import word_tokenize
import os.path as path
import re
import time
from nltk.stem import SnowballStemmer
from bm25_md import BM25_Model
from nltk.corpus import stopwords
stop_english = set(stopwords.words('english'))

# filename_ls = ['data/QA_train.json']
filename_ls = ['data/QA_dev.json']
dataset = []
t0 = time.time()
parent_path = path.dirname(__file__)
# path.join(pp, filename)
parent_path = path.dirname(parent_path)
print parent_path
# exit()
stemmer = SnowballStemmer("english")


def my_tokenize(sentence):
    """ This the is tokenize function, part of the feature engineering """
    sentence = sentence.lower()
    ll = word_tokenize(sentence)
    lls = [stemmer.stem(ii) for ii in ll if re.search(r'[a-z0-9]+', ii) and ii not in stop_english]
    return lls


kii = 0
total_i = 0


def build_model_and_evaluate(clf, X, y, report=False):
    global kii, total_i
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

    kii += ac_i
    total_i += len(y)
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

csvv = open('bm25_dev.csv', mode='w', )
fieldnames = ['document_i', 'question_i', 'prediction_i',
              'actual_yi', 'correctness', 'question', 'prediction_sentence', 'actual_y_sentence', 'answer']
writer = csv.DictWriter(csvv, fieldnames=fieldnames, )
writer.writeheader()

ddi = 0
for col in dataset:

    qX = [i['question'] for i in col['qa']]
    ans_y = [i['answer'] for i in col['qa']]
    qy = [i['answer_sentence'] for i in col['qa']]
    model = col['model']
    table, acc = build_model_and_evaluate(model, qX, qy)
    for t in table:
        t['document_i'] = ddi
        t['question'] = qX[t['question_i']].encode('utf-8')
        t['answer'] = ans_y[t['question_i']].encode('utf-8')
        # psss =
        # t['prediction_sentence'] = str(col['sentences'][t['prediction_i'][0]]).encode('utf-8')
        t['actual_y_sentence'] = col['sentences'][t['actual_yi']].encode('utf-8').decode('utf-8')
        writer.writerow(t)

    ddi += 1




print 'EXEC:', time.time() - t0

print 'Finally................ accuracy:', kii/total_i

