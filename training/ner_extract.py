import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath((inspect.getfile(inspect.currentframe()))))
parentdir = os.path.dirname(currentdir)
from nltk import word_tokenize
# sys.path.insert(0, parentdir)
# print parentdir
# exit()
import json
import time
from nltk import StanfordNERTagger
import re


def json_load_byteified(file_handle):
    return _byteify(json.load(file_handle, object_hook=_byteify),ignore_dicts=True)

def _byteify(data, ignore_dicts = False):
    # if this is a unicode string, return its string representation
    if isinstance(data, unicode):
        return data.encode('utf-8')
    # if this is a list of values, return list of byteified values
    if isinstance(data, list):
        return [ _byteify(item, ignore_dicts=True) for item in data ]
    # if this is a dictionary, return dictionary of byteified keys and values
    # but only if we haven't already byteified it
    if isinstance(data, dict) and not ignore_dicts:
        return {
            _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
            for key, value in data.iteritems()
        }
    # if it's anything else, return it in its original form
    return data

with open(os.path.join(parentdir, "data/QA_dev.json")) as json_file:
    json_data = json_load_byteified(json_file)
print "import success"


import os
# java_path = "C:/Program Files/Java" # replace this
# os.environ['JAVAHOME'] = java_path

cwd = os.getcwd()
st = StanfordNERTagger(os.path.join(parentdir, 'data/english.all.3class.distsim.crf.ser.gz')
                       , os.path.join(parentdir, 'data/stanford-ner.jar'))


output_file = "dev_ner2.json"

if not os.path.isfile(output_file):
    start = time.time()
    progressT = len(json_data)
    listOfDocument = []
    i = 0
    for jd in json_data:
        aList = []
        aList.extend([st.tag_sents(
            [word_tokenize(re.sub(',', '', re.sub('[^a-zA-Z0-9-_*., ]', ' ', x['question']))) for x in jd['qa']])])
        aList.extend([st.tag_sents(
            [word_tokenize(re.sub(',', '', re.sub('[^a-zA-Z0-9-_*., ]', ' ', x['answer']))) for x in jd['qa']])])
        aList.extend([st.tag_sents(
            [word_tokenize(re.sub(',', '', re.sub('[^a-zA-Z0-9-_*., ]', ' ', x))) for x in jd['sentences']])])
        if len(listOfDocument) == 0:
            listOfDocument.append(aList)
        else:
            listOfDocument.extend(aList)
        i += 1
        sys.stdout.write('\r')
        sys.stdout.write("%d%%" % (i * 100 / progressT))
        sys.stdout.flush()

    with open(output_file, 'w') as outfile:
        json.dump(listOfDocument, outfile)
    end = time.time()
    print '\nTime spending:', end - start
else:
    print 'there is a file'


# with open(output_file) as json_file:
#     json_dataNER = json_load_byteified(json_file)