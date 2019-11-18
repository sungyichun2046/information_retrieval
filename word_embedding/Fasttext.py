# -*- coding: utf-8 -*-
from __future__ import absolute_import
import sys
import pandas as pd
sys.path.append('/home/yichun/projects/information_retrieval')
import fasttext
"""
USAGE:
vectorizer = Fasttext()
vectorizer.create_training_file()
vectorizer.train()
"""

class Fasttext():
    """
        Build a fasttext vectorizer using library fasttext
    """
    def create_training_file(self):
        judgements = pd.read_csv('../data/judgements')
        texts = list(judgements['text'])
        for text in texts:
            with open('data/corpus.txt', 'a') as f:
                f.write("{}{}".format(text, '\n'))

    def train(self):
        model = fasttext.train_unsupervised("data/corpus.txt", model='cbow', minCount=10, lr=0.05, dim=100, ws=5, epoch=5)
        model.save_model("fasttext_model.bin")






