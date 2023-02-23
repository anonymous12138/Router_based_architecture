import pickle as pkl
import os
import re
import pandas as pd
from datetime import datetime
import numpy as np
import string
import logging

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

from sklearn.model_selection import train_test_split

import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Activation, Conv2D, Input, Embedding, Reshape, MaxPool2D, Concatenate, Flatten, Dropout, Dense, Conv1D, MaxPooling1D, TimeDistributed
from tensorflow.keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, SpatialDropout1D, Layer, Embedding, Bidirectional, GRU, SpatialDropout2D
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate,dot,add,subtract,multiply
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras import regularizers, constraints
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, LSTM, Embedding,Bidirectional

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
        self.word_index = None
        self.tokenizer = None
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

    def create_embedding(self, path, data, name, split, preprocess = True):
        if preprocess:
            # Word Embedding
            embeddings_index = dict()
            f = open(path + "glove.twitter.27B.200d.txt")
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            f.close()
            print('Loaded %s word vectors.' % len(embeddings_index))

            EMBED_SIZE = 200

            min_wordCount = 1
            absent_words = 0
            small_words = 0
            embedding_matrix = np.zeros((len(self.word_index) + 1, self.EMBED_SIZE))
            word_counts = self.tokenizer.word_counts
            for word, i in self.word_index.items():
                if word_counts[word] >= min_wordCount:
                    embedding_vector = embeddings_index.get(word)
                    if embedding_vector is not None:
                        # words not found in embedding index will be all-zeros.
                        embedding_matrix[i] = embedding_vector
                    else:
                        absent_words += 1
                else:
                    small_words += 1
            print('Total absent words are', absent_words, 'which is', "%0.2f" % (absent_words * 100 / len(self.word_index)),
                  '% of total words')
            print('Words with ' + str(min_wordCount) + ' or less mentions', small_words, 'which is',
                  "%0.2f" % (small_words * 100 / len(self.word_index)),
                  '% of total words')
            print(str(len(self.word_index) - small_words - absent_words) + ' words to proceed.')
            with (open(path + name + '_embedding' + split + '.pkl', "wb")) as openfile:
                pkl.dump(embedding_matrix, openfile, protocol=pkl.HIGHEST_PROTOCOL)
        else:
            with (open(path + name + '_embedding' + split + '.pkl', "rb")) as openfile:
                embedding_matrix = pkl.load(openfile)
        return embedding_matrix


def tokenize_data(self, data):
    self.tokenizer = keras.preprocessing.text.Tokenizer(num_words=self.MAX_NB_WORDS, lower=True, oov_token=None)
    self.tokenizer.fit_on_texts(data.text.values)
    self.word_index = self.tokenizer.word_index
    return self.word_index


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


def convert_embedding(self, data):
    y_data = data.emotions
    X_data = data.drop(['emotions'], axis=1)
    _y_data = [self.classes[t] for t in y_data]
    _X_data = np.zeros((len(X_data), self.MAX_SENT_LENGTH), dtype='int32')

    for i, post in enumerate(X_data.text):
        wordTokens = text_to_word_sequence(post)
        j = 0
        for _, word in enumerate(wordTokens):
            if j < self.MAX_SENT_LENGTH and self.tokenizer.word_index[word] < self.MAX_NB_WORDS:
                _X_data[i, j] = self.tokenizer.word_index[word]
                j += 1
    _y_data = to_categorical(np.asarray(_y_data))
    return _X_data, _y_data


def train(self, _X_train, _y_train):
    class_num = _y_train.shape[1]

    model = Sequential()
    model.add(Embedding(input_dim=len(self.word_index) + 1,
                        output_dim=self.EMBEDDING_DIM,
                        input_length=_X_train.shape[1],
                        weights=[self.embedding_matrix], trainable=True))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(200, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(200, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(200, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(200, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(200, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(200, return_sequences=True)))
    model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(100, return_sequences=False)))
    model.add(Dense(class_num, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5)
    mc = ModelCheckpoint('/model_complex_new.h5',
                         monitor='val_accuracy',
                         mode='max',
                         verbose=1,
                         save_best_only=True)
    history = model.fit(_X_train,
                        _y_train,
                        validation_split=0.01,
                        epochs=25,
                        batch_size=1024,
                        verbose=1,
                        callbacks=[es, mc])

    return model


def load_model(self, path):
    model = keras.models.load_model(path + 'model_complex.h5')
    return model


def prepare_data(self, path, data, name, split, preprocess=True):
    processed_data = self.process_data(path, data, name, split, preprocess)
    tokenized_data = self.tokenize_data(data)
    X_data, y_data = self.convert_embedding(data)
    return X_data, y_data


def predict(self, clf, X_test):
    y_pred = np.argmax(self.model.predict(X_test), axis=1)
    y_proba = self.model.predict_proba(X_test)
    return y_pred, y_proba
