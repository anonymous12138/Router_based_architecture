import pandas as pd
import numpy as np
import os
import copy
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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from transformers import AutoConfig, TFAutoModelForTokenClassification
from transformers import DistilBertTokenizer, RobertaTokenizer
from transformers import AutoTokenizer, AutoModelForTokenClassification

import os
import itertools
import pandas as pd
import numpy as np
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification, EarlyStoppingCallback
import torch
from transformers import pipeline

from datasets import Dataset

from collections import Counter
from conlleval import evaluate
from datasets import load_dataset


class BERT_model(object):
    def __init__(self, data_loc, model_checkpoint, task, batch_size):
        self.split_train = self.split_text_label(os.path.join(data_loc, "train.txt"))
        self.split_validation = self.split_text_label(os.path.join(data_loc, "valid.txt"))
        self.split_test = self.split_text_label(os.path.join(data_loc, "test.txt"))
        #         with open('sentences_df.pkl', 'rb') as handle:
        #             sentences_df = pkl.load(handle)

        #         train_eval_set = sentences_df.sample(frac=0.80,random_state=200) #random state is a seed value
        #         self.split_test = sentences_df.drop(train_eval_set.index)
        #         self.split_train = train_eval_set.sample(frac=0.90,random_state=200)
        #         self.split_validation = train_eval_set.drop(self.split_train.index)

        self.model_checkpoint = model_checkpoint
        self.task = task
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

        self.label_list = ['B-LOC', 'B-MISC', 'B-ORG',
                           'B-PER', 'I-LOC', 'I-MISC',
                           'I-ORG', 'I-PER', 'O']
        self.label_encoding_dict = {
            'B-LOC': 0, 'B-MISC': 1, 'B-ORG': 2,
            'B-PER': 3, 'I-LOC': 4, 'I-MISC': 5,
            'I-ORG': 6, 'I-PER': 7, 'O': 8
        }

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

    def data_process_BERT(self, df):
        sentences = []
        ner_tags = []
        for sent in df['Sentence #'].unique():
            sentence = []
            tag = []
            words = df[df['Sentence #'] == sent]
            words.reset_index(inplace=True, drop=True)
            sentence = list(words.to_dict()['Word'].values())
            tag = list(words.to_dict()['Tag'].values())
            sentences.append(sentence)
            ner_tags.append(tag)
        sentences_df = pd.DataFrame()
        sentences_df['tokens'] = sentences
        sentences_df['ner_tags'] = ner_tags
        sentences_df.head()
        return sentences_df

    def tokenize_and_align_labels(self, examples):
        label_all_tokens = True
        tokenized_inputs = self.tokenizer(list(examples["tokens"]), truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"{self.task}_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif label[word_idx] == '0':
                    label_ids.append(0)
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label_encoding_dict[label[word_idx]])
                else:
                    label_ids.append(self.label_encoding_dict[label[word_idx]] if label_all_tokens else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def prepare_data(self, data):
        split_df = self.create_data_frame(data)
        split_df.reset_index(inplace=True, drop=True)
        data_set = self.data_process_BERT(split_df)
        dataset = Dataset.from_pandas(data_set)
        tokenized_datasets = dataset.map(self.tokenize_and_align_labels, batched=True)
        return tokenized_datasets

    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [[self.label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in
                            zip(predictions, labels)]
        true_labels = [[self.label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in
                       zip(predictions, labels)]

        results = metrics.compute(predictions=true_predictions, references=true_labels)
        flattened_results = {"overall_precision": results["overall_precision"],
                             "overall_recall": results["overall_recall"],
                             "overall_f1": results["overall_f1"],
                             "overall_accuracy": results["overall_accuracy"]
                             }
        for k in results.keys():
            if (k not in flattened_results.keys()):
                flattened_results[k + "_f1"] = results[k]["f1"]

        return flattened_results

    def train(self, train_tokenized_datasets, eval_tokenized_datasets):
        model = AutoModelForTokenClassification.from_pretrained(self.model_checkpoint,
                                                                num_labels=len(self.label_list))
        args = TrainingArguments(
            f"test-{self.task}",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=1e-4,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=300,
            weight_decay=1e-5,
            load_best_model_at_end=True
        )
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        metric = load_metric("seqeval")

        early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                         patience=5,
                                                         restore_best_weights=True)
        print(train_tokenized_datasets)
        trainer = Trainer(
            model,
            args,
            train_dataset=train_tokenized_datasets,
            eval_dataset=eval_tokenized_datasets,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )

        trainer.train()
        trainer.evaluate()
        return trainer

    def save_model(self, trainer, loc='Data/NER/dslim_model.model'):
        trainer.save_model(loc)

    def load_model(self, loc):
        model = AutoModelForTokenClassification.from_pretrained(loc,
                                                                num_labels=len(self.label_list))
        return model

    def predict(self, trainer, test_tokenized_datasets):
        predicted = trainer.evaluate(test_tokenized_datasets)
        return predicted

