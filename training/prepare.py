import json
from nltk.tokenize import word_tokenize
import os.path as path
import time
from nltk.corpus import stopwords
stop_english = set(stopwords.words('english'))
import sys

cur_path = path.dirname(__file__)
parent_path = path.dirname(cur_path)
print parent_path

t0 = time.time()
datas = 'data/QA_dev.json'
print datas


from nltk import StanfordNERTagger, StanfordPOSTagger

dataset = json.loads(open(path.join(parent_path, datas)).readline())

ner_tagger = StanfordNERTagger(path.join(parent_path, 'data/english.all.3class.distsim.crf.ser.gz'),
                               path.join(parent_path, 'data/stanford-ner.jar'),
                               encoding='utf-8')

pos_tagger = StanfordPOSTagger(path.join(parent_path, 'data/wsj-0-18-left3words-distsim.tagger'),
                               path.join(parent_path, 'data/stanford-postagger.jar'),
                               encoding='utf-8')

prog_total = len(dataset)


def dmerge(ner, pos):
    if pos and pos[1] == 'CD':
        return ner[0], 'NUMBER'
    elif ner[1] == 'O':
        return pos
    else:
        return ner


def _merge_tag(ners, poss):
    return map(dmerge, ners, poss)


def putq(t, dic):
    dic['question'] = t


def puta(t, dic):
    dic['answer'] = t


def put_asent(t, dic):
    dic['ans_sent'] = t
# error happens some how for #68, #97, #209, #263, #329,

# error happens some how for #68, #97, #209, #263, #329,

# for i in [68,97,209,263,329]:
#     col = dataset[i]
#     for qa in col['qa']:
#         qa['ans_sent'] = col['sentences'][qa['answer_sentence']].encode('utf-8')
#         qa['question'] = word_tokenize(qa['question'])
#         qa['answer'] = word_tokenize(qa['answer'])
#         qa['ans_sent'] = word_tokenize(qa['ans_sent'].decode('utf-8'))
#     pass
#
#     questions_ner = ner_tagger.tag_sents([q['question'] for q in col['qa']])
#     questions_pos = pos_tagger.tag_sents([q['question'] for q in col['qa']])
#     q_tags = map(_merge_tag, questions_ner, questions_pos)
#     map(putq, q_tags, col['qa'])
#
#     answer_ner = ner_tagger.tag_sents([q['answer'] for q in col['qa']])
#     answer_pos = pos_tagger.tag_sents([q['answer'] for q in col['qa']])
#     a_tags = map(_merge_tag, answer_ner, answer_pos)
#     map(puta, a_tags, col['qa'])
#
#     asent_ner = ner_tagger.tag_sents([q['ans_sent'] for q in col['qa']])
#     asent_pos = pos_tagger.tag_sents([q['ans_sent'] for q in col['qa']])
#     asent_tags = map(_merge_tag, asent_ner, asent_pos)
#     map(put_asent, asent_tags, col['qa'])
#
#
# exit()
filename = 'xdev_cb.json'
jfile = open(filename, 'w')
prog_i = 0.0
for col in dataset:
    for qa in col['qa']:
        qa['ans_sent'] = col['sentences'][qa['answer_sentence']].encode('utf-8')
        qa['question_tks'] = word_tokenize(qa['question'])
        qa['answer'] = word_tokenize(qa['answer'])
        qa['ans_sent_tks'] = word_tokenize(qa['ans_sent'].decode('utf-8'))

    try:
        questions_ner = ner_tagger.tag_sents([q['question_tks'] for q in col['qa']])
        questions_pos = pos_tagger.tag_sents([q['question_tks'] for q in col['qa']])
        q_tags = map(_merge_tag, questions_ner, questions_pos)
        map(putq, q_tags, col['qa'])

        answer_ner = ner_tagger.tag_sents([q['answer'] for q in col['qa']])
        answer_pos = pos_tagger.tag_sents([q['answer'] for q in col['qa']])
        a_tags = map(_merge_tag, answer_ner, answer_pos)
        map(puta, a_tags, col['qa'])

        asent_ner = ner_tagger.tag_sents([q['ans_sent_tks'] for q in col['qa']])
        asent_pos = pos_tagger.tag_sents([q['ans_sent_tks'] for q in col['qa']])
        asent_tags = map(_merge_tag, asent_ner, asent_pos)
        map(put_asent, asent_tags, col['qa'])
    except:
        col['qa'] = None
        col = None
        print 'Error:', prog_i, '#####'


    if col:
        jfile.write(json.dumps(col) + '\n')
        del col['sentences']
    prog_i += 1
    sys.stdout.write('\r')
    sys.stdout.write("%f%%" % (prog_i * 100.0 / prog_total))
    sys.stdout.flush()

ccc = 0




# open('xtrain_ner_pp.json', 'w').write(json.dumps(dataset, indent=4))


print 'EXEC:', time.time() - t0
# error happens some how for #68, #97, #209, #263, #329,
# error happens some how for #68, #97, #209, #263, #329,
# error happens some how for #68, #97, #209, #263, #329,