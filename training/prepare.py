import json
from nltk.tokenize import word_tokenize
import os.path as path
import re
import time
import sys
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
stop_english = set(stopwords.words('english'))

# filename_ls = ['data/QA_train.json']
filename_ls = ['data/QA_dev.json']
t0 = time.time()
cur_path = path.dirname(__file__)
# path.join(pp, filename)
parent_path = path.dirname(cur_path)
print parent_path

from nltk import StanfordNERTagger, StanfordPOSTagger

dataset = json.loads(open(path.join(parent_path, 'data/QA_train.json')).readline())

ner_tagger = StanfordNERTagger(path.join(parent_path, 'data/english.all.3class.distsim.crf.ser.gz'),
                               path.join(parent_path, 'data/stanford-ner.jar'), encoding='utf-8')

pos_tagger = StanfordPOSTagger(path.join(parent_path, 'data/english-bidirectional-distsim.tagger'),
                               path.join(parent_path, 'data/stanford-postagger.jar'), encoding='utf-8')

total = len(dataset)

for col in dataset:
    for qa in col['qa']:
        qa['anss'] = col['sentences'][qa['answer_sentence']].encode('utf-8')
        qa['question'] = word_tokenize(qa['question'].encode('utf-8'))
        qa['answer'] = word_tokenize(qa['answer'])
        qa['anss'] = word_tokenize(qa['anss'].decode('utf-8'))

    questions_pos = ner_tagger.tag_sents([q['question'] for q in col['qa']])
    qq = 0
    # sys.stdout.write('\r')
    # sys.stdout.write("%d%%" % (i * 100 / progressT))
    # sys.stdout.flush()
    del col['sentences']
ccc =0

