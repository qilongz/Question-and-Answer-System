import csv
from nltk import FreqDist
# from nltk.tokenize import TweetTokenizer
import re
from numpy import random
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords as sw
import json
import pickle


class SentimentAnalysis:
    def __init__(self, loadsaved=True):
        self.classifier = None
        # if loadsaved:
        #     try:
        #         fs = open('sentimentAPI.model', mode='rb')
        #         ss = fs.readlines()
        #         self.classifier = pickle.loads(ss)
        #         return
        #     except Exception:
        #         print('Failed')
        #         pass
        self.dataset = None
        self.stopwords = sw.words('english')
        self.dset_positive = []
        self.dset_negative = []
        self.dset_neutral = []
        self.train_X, self.train_y = None, None
        self.dev_X,  self.dev_y = None, None
        self.test_X, self.test_y = None, None

        # self.tokenizer_tw = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
        self.word_freq_vectorizer = DictVectorizer()

        self.read_dataset()
        self.process_make_matrix()
        self.classifier = self.build_model_and_evaluate(RandomForestClassifier())

    def read_dataset(self, filename_ls=('data/QA_train.json',)):
        self.dataset = []
        for ff in filename_ls:
            file_strs = open(ff).readline()
            self.dataset += json.loads(file_strs)


    # def tokenize_process(self, ls, tag=None):
    #     new_lls = []
    #     for lt in ls:
    #         tokens = self.tokenizer_tw.tokenize(lt[1])
    #         tokens = [i for i in tokens if not re.search(r'http://', i) and i not in self.stopwords]
    #
    #         bag_of_word_dict = FreqDist(tokens)
    #         new_lls.append((bag_of_word_dict, lt[0]))
    #         assert lt[0] == tag
    #     return new_lls
    #
    # def segmentation(self, data):
    #     # print data[0], 'AA'
    #     random.shuffle(data)
    #     # print data[0], 'BB'
    #     dlen = len(data)
    #     train, dev, test = data[:int(0.8 * dlen)], data[int(0.8 * dlen): int(0.9 * dlen)], data[int(0.9 * dlen):]
    #     return train, dev, test

    def process_make_matrix(self):
        # self.dset_positive = self.tokenize_process(self.dset_positive, 'positive')
        # self.dset_negative = self.tokenize_process(self.dset_negative, 'negative')
        # self.dset_neutral = self.tokenize_process(self.dset_neutral, 'neutral')
        #
        # ptrain, pdev, ptest = self.segmentation(self.dset_positive)
        # ntrain, ndev, ntest = self.segmentation(self.dset_negative)
        # xtrain, xdev, xtest = self.segmentation(self.dset_neutral)

        # train_raw = ptrain + ntrain + xtrain
        # random.shuffle(train_raw)
        # train_x_raw = [x[0] for x in train_raw]
        # self.train_X = self.word_freq_vectorizer.fit_transform(train_x_raw)
        # self.train_y = [x[1] for x in train_raw]
        #
        # dev_raw = pdev + ndev + xdev
        # random.shuffle(dev_raw)
        # dev_x_raw = [x[0] for x in dev_raw]
        # self.dev_X = self.word_freq_vectorizer.transform(dev_x_raw)
        # self.dev_y = [x[1] for x in dev_raw]
        #
        # test_raw = ptest + ntest + xtest
        # random.shuffle(test_raw)
        # test_x_raw = [x[0] for x in test_raw]
        # self.test_X = self.word_freq_vectorizer.transform(test_x_raw)
        # self.test_y = [x[1] for x in test_raw]

    def build_model_and_evaluate(self, clf, report=True):
        # training
        clf.fit(self.train_X, self.train_y)
        # test
        pred = clf.predict(self.test_X)
        # score
        accuracy = metrics.accuracy_score(self.test_y, pred)
        if report:
            print('-' * 100)
            print('classifier:')
            print(clf)

            print("macro f1 score:   %0.3f" % metrics.f1_score(self.test_y, pred, average='macro'))
            print("accuracy:   %0.3f" % accuracy, '\n\n')
            print(metrics.classification_report(self.test_y, pred))
            print()
            print(metrics.confusion_matrix)

        print()

        # save model:
        # ss = pickle.dumps(clf)
        # fs = open('sentimentAPI.model', mode='wb')
        # fs.write(ss)
        # fs.close()

        return clf

    def sentiment_analysis(self, tweet_text):
        pred = None
        if isinstance(tweet_text, str):
            tokens = self.tokenizer_tw.tokenize(tweet_text)
            tokens = [i for i in tokens if not re.search(r'http://', i) and i not in self.stopwords]
            bag_of_word_dict = FreqDist(tokens)
            tweet_vector = self.word_freq_vectorizer.transform(bag_of_word_dict)
            pred = self.classifier.predict(tweet_vector)
        elif isinstance(tweet_text, (list, tuple)):
            pred = []
            for tt in tweet_text:
                tokens = self.tokenizer_tw.tokenize(tt)
                tokens = [i for i in tokens if not re.search(r'http://', i) and i not in self.stopwords]
                bag_of_word_dict = FreqDist(tokens)
                tweet_vector = self.word_freq_vectorizer.transform(bag_of_word_dict)
                pred += self.classifier.predict(tweet_vector)

        return pred

