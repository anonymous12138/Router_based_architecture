import textstat
import pandas as pd
import matplotlib
matplotlib.get_backend()
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()


class Router(object):
    def __init__(self):
        self.easy_model = None
        self.hard_model = None
        self.router = None

    def entropy(self, arr):
        arr = arr[:(arr != 0).argmin()]  # only keep the non-zero part
        size = len(arr)
        unique = np.unique(arr)
        p = []
        for i in range(len(unique)):
            p.append((np.count_nonzero(arr == unique[i])) / size)
        ent = 0
        for i in range(len(p)):
            ent += (-p[i] * (np.log2(p[i])))
        return ent

    def load_easy_model(self, filename):
        if not self.easy_model:
            self.easy_model = pickle.load(open(filename, 'wb'))
        else:
            print("Model already loaded!")
        return

    def load_hard_model(self, filename):
        if not self.hard_model:
            self.hard_model = pickle.load(open(filename, 'wb'))
        else:
            print("Model already loaded!")
        return

    def save_router(self, filename):
        if self.router:
            pickle.dump(self.router, open(filename, 'wb'))
        else:
            print("Router does not exist!")
        return

    def fit(self, X_train, y_train):
        """
        X_train: n-d nparray
        y_train: 1-d nparray
        """
        self.router = DecisionTreeClassifier()
        y_pred = self.easy_model.predict(X_train)
        # compute probability
        y_proba = self.easy_model.predict_proba(X_train)

        # compute correctness
        choice = []
        for i in range(len(y_pred)):
            if y_pred[i] == y_train[i]:
                choice.append(1)
            else:
                choice.append(0)
        # compute entropy
        ent = []
        for i in range(len(X_train)):
            ent.append(self.entropy(X_train[i]))
        # compute readability
        readability = []
        for i in range(len(X_train)):
            readability.append(textstat.dale_chall_readability_score(X_train[i]))
        # combine all scores
        X_train_combined = pd.concat([pd.DataFrame(ent), readability, y_proba], axis=1)
        self.router.fit(X_train_combined, choice)
        return

    def predict(self, X_test, y_test, classes):
        y_proba = self.easy_model.predict_proba(X_test)
        ent = []
        for i in range(len(X_test)):
            ent.append(self.entropy(X_test[i]))
        readability = []
        for i in range(len(X_test)):
            readability.append(textstat.dale_chall_readability_score(X_test[i]))
        X_test_combined = pd.concat([pd.DataFrame(ent), readability, y_proba], axis=1)
        y_choice = self.router.predict(X_test_combined)
        y_final = []
        for i in range(len(y_choice)):
            if y_choice[i] == 1:
                y_final.append(self.easy_model.predict([X_test[i]])[0])
            else:
                y_final.append(self.hard_model.predict([X_test[i]])[0])
        return