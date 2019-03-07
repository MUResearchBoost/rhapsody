from enum import Enum
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups

def to_one_hot(y, n_class):
    return np.eye(n_class)[y.astype(int)]


def load_data(file_name, sample_ratio, n_class, names, one_hot=True):
    '''load data from .csv file'''
    csv_file = pd.read_csv(file_name, names=names)
    shuffle_csv = csv_file.sample(frac=sample_ratio)
    x = pd.Series(shuffle_csv["content"])
    y = pd.Series(shuffle_csv["class"])
    if one_hot:
        y = to_one_hot(y, n_class)
    return x, y

def print_top_words(model, feature_names, n_top_words):
    words = dict()
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        print(message)
        words[message] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]

    print()
    return words

class dataset:
    set_choice = ['dbpedia', 'homemade']
    def tutorial_dataset(self, name ,set, sample_ratio):
        if name == 'dbpedia':
            address = "./dbpedia_data/dbpedia_csv/train.csv"
            names = ["class", "title", "content"]
            if set == 'train':
                x_train, y_train = load_data(address, sample_ratio, 15, names, one_hot=False)
                return x_train, y_train
            elif set == 'test':
                x_test, y_test = load_data(address, sample_ratio, 15, names, one_hot=False)
                return x_test, y_test
        if name == 'fetch_20newsgroup':
            begin = 0
            end = 1000
            xx = fetch_20newsgroups(shuffle=True, random_state=1,
                                         remove=('headers', 'footers', 'quotes'))
            x = xx.data[begin:end]
            y = xx.target[begin:end]
            return x, y

    def  homemade(self, set, sample_ratio):
        x_train = []
        y_train = []
        return  x_train, y_train







