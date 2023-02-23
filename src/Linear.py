import pickle as pkl
import os
import re
import pandas as pd
from datetime import datetime
import numpy as np
import string

from sklearn.model_selection import train_test_split
import logging

import keras
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer

from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pickle, re
import numpy as np
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download('punkt')
nltk.download('wordnet')

import textstat
nltk.download('punkt')
nltk.download('stopwords')
stemmer = PorterStemmer()


class linear_emotion_detection(object):
    def __init__(self):
        self.DATA_COLUMN = 'text'
        self.LABEL_COLUMN = 'emotions'
        self.label_list = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

        self.MAX_SENTENCE_NUM = 128
        self.MAX_WORD_NUM = 128
        self.MAX_FEATURES = 200000

        self.MAX_SENT_LENGTH = 128
        self.MAX_SENTS_body = 30
        self.MAX_SENTS_header = 2
        self.MAX_NB_WORDS = 200000
        self.EMBEDDING_DIM = 200
        self.VALIDATION_SPLIT = 0.2
        self.vectorizer = None
        self.classes = {'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5}

    def cleanString(self, review, stopWords):
        returnString = ""
        sentence_token = sent_tokenize(review)
        idx_list = []
        return sentence_token, idx_list

    def load_data(self, data_loc):
        with (open(data_loc, "rb")) as openfile:
            df = pkl.load(openfile)
        df.reset_index(inplace=True, drop=True)
        return df

    def process_data(self, path, df, name, split, preprocess=True):
        if preprocess:
            df_new = pd.DataFrame([], columns=['text', 'emotions'])
            for i in range(df.shape[0]):
                text = df.loc[i, 'text']
                text = ' '.join([word.strip(string.punctuation) for word in text.split() if
                                 word.strip(string.punctuation) is not ""])
                text, _ = self.cleanString(text, stopwords.words("english"))
                df_new.loc[i, 'text'] = text[0]
                df_new.loc[i, 'emotions'] = df.loc[i, 'emotions']
                if i % 10000 == 0:
                    print(i)
                with (open(path + '/' + name + '_processed_' + split + '.pkl', "wb")) as openfile:
                    pkl.dump(df_new, openfile, protocol=pkl.HIGHEST_PROTOCOL)
        else:
            with (open(path + '/' + name + '_processed_' + split + '.pkl', "rb")) as openfile:
                df_new = pkl.load(openfile)
        return df_new

    def tokenize_data(self, data):
        self.tokenizer = keras.preprocessing.text.Tokenizer(num_words=self.MAX_NB_WORDS, lower=True, oov_token=None)
        self.tokenizer.fit_on_texts(data.text.values)
        word_index = self.tokenizer.word_index
        return word_index

    def divide_data(self, data):
        y = data.emotions
        X = data.drop(['emotions'], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        train_df = X_train
        train_df['emotions'] = y_train
        train_df.reset_index(inplace=True, drop=True)
        test_df = X_test
        test_df['emotions'] = y_test
        test_df.reset_index(inplace=True, drop=True)

        new_train_df = pd.DataFrame()
        for cla in y_train.unique():
            sub_df = train_df[train_df['emotions'] == cla]
            sub_df.reset_index(inplace=True, drop=True)
            new_train_df = pd.concat([new_train_df, sub_df.loc[0:3000]], axis=0)

        new_train_df.reset_index(inplace=True, drop=True)
        return new_train_df, test_df

    def TF_IDF(self, data):
        y_data = data.emotions
        X_data = data.drop(['emotions'], axis=1)
        self.vectorizer = TfidfVectorizer(min_df=5,
                                          max_df=0.8,
                                          sublinear_tf=True,
                                          use_idf=True)
        classes = {'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5}
        train_vectors = self.vectorizer.fit_transform(X_data['text'])
        _y_data = [classes[t] for t in y_data]
        return train_vectors, _y_data

    def train(self, train_vectors, _y_train):
        clf = SVC(probability=True)
        clf.fit(train_vectors, _y_train)
        return clf

    def prepare_data(self, path, data, name, split, preprocess=True):
        processed_data = self.process_data(path, data, name, split, preprocess)
        tokenized_data = self.tokenize_data(data)
        X_data, y_data = self.TF_IDF(data)
        return X_data, y_data

    def predict(self, clf, X_test):
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        return y_pred, y_proba

    def save_model(self, clf):
        filename = 'Data/Sentiment_Analysis/Linear_model.sav'
        pkl.dump(clf, open(filename, 'wb'))

    def load_model(self, filename):
        model = pkl.load(open(filename, 'rb'))
        return model