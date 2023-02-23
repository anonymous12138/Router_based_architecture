import pandas as pd
import numpy as np
import os
import pickle as pkl

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from collections import Counter


class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s['Word'].values.tolist(),
                                                     s['Tag'].values.tolist())]
        self.grouped = self.data.groupby('Sentence #').apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


class CRF_model(object):
    def __init__(self, data_loc):
        self.split_train = self.split_text_label(os.path.join(data_loc, "train.txt"))
        self.split_validation = self.split_text_label(os.path.join(data_loc, "valid.txt"))
        self.split_test = self.split_text_label(os.path.join(data_loc, "test.txt"))

    def split_text_label(self, filename):
        f = open(filename)
        split_labeled_text = []
        sentence = []
        for line in f:
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                if len(sentence) > 0:
                    split_labeled_text.append(sentence)
                    sentence = []
                continue
            splits = line.split(' ')
            sentence.append([splits[0], splits[-1].rstrip("\n")])
        if len(sentence) > 0:
            split_labeled_text.append(sentence)
            sentence = []
        return split_labeled_text

    def create_data_frame(self, data):
        for i in range(len(data)):
            sent = 'Sent_' + str(i)
            for j in range(len(data[i])):
                data[i][j].append(sent)
        flat_split_data = [item for sublist in data for item in sublist]
        split_df = pd.DataFrame(flat_split_data, columns=['Word', 'Tag', 'Sentence #'])
        return split_df

    def word2features(self, sent, i):
        word = sent[i][0]
        postag = sent[i][1]

        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[:2],
        }
        if i > 0:
            word1 = sent[i - 1][0]
            postag1 = sent[i - 1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True
        if i < len(sent) - 1:
            word1 = sent[i + 1][0]
            postag1 = sent[i + 1][1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True
        return features

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def sent2labels(self, sent):
        return [label for token, label in sent]

    def sent2tokens(self, sent):
        return [token for token, label in sent]

    def prepare_data(self, data):
        split_df = self.create_data_frame(data)
        data_getter = SentenceGetter(split_df)
        data_sentences = data_getter.sentences
        return split_df, data_getter, data_sentences

    def train(self):
        split_train_df, train_getter, train_sentences = self.prepare_data(self.split_train)
        split_validation_df, validation_getter, validation_sentences = self.prepare_data(self.split_validation)

        X_train = [self.sent2features(s) for s in train_sentences]
        y_train = [self.sent2labels(s) for s in train_sentences]

        X_validation = [self.sent2features(s) for s in validation_sentences]
        y_validation = [self.sent2labels(s) for s in validation_sentences]

        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        crf.fit(X_train, y_train)

        y = split_train_df.Tag.values
        classes = np.unique(y)
        classes = classes.tolist()
        new_classes = classes.copy()
        new_classes.pop()
        return crf, new_classes

    def save_model(self, crf):
        filename = 'Data/NER/CRF_model.sav'
        pkl.dump(crf, open(filename, 'wb'))

    def load_model(self, filename):
        crf = pkl.load(open(filename, 'wb'))
        return crf

    def predict(self, crf, X_test, y_test, classes):
        y_pred = crf.predict(X_test)
        return y_pred
