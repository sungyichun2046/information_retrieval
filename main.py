from __future__ import absolute_import
import sys
import pandas as pd
sys.path.append('/home/yichun/projects/information_retrieval')
import os.path
from process import store_decomposed_sentences_df

if not os.path.isfile('information_retrieval/data/judgements'):
    store_decomposed_sentences_df('information_retrieval/data/judgements.jsonl')
else:
    judgements = pd.read_csv('information_retrieval/data/judgements')


