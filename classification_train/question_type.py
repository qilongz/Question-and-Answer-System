import json
import os
from nltk.tokenize import word_tokenize
import os.path as path
import re
import time
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
stop_english = set(stopwords.words('english'))

# filename_ls = ['data/QA_train.json']
filename_ls = ['data/QA_train.json']
t0 = time.time()
parent_path = path.dirname(__file__)
# path.join(pp, filename)
parent_path = path.dirname(parent_path)

dataset = json.loads(open(path.join(parent_path, 'data/NERtrain.json')).readline())

questions
xxx = 0
