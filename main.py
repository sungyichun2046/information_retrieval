from __future__ import absolute_import
import sys
import pandas as pd
import json
sys.path.append('/home/yichun/projects/information_retrieval')
from process import store_decomposed_sentences_df
from compute_similarity.sort import get_judgements_sorted
from extractor_keywords import get_extracted_keywords

CONFIG = json.load(open("information_retrieval/config.json"))
#store_decomposed_sentences_df('information_retrieval/data/judgements.jsonl') # uncomment this line the first time
judgements = pd.read_csv('information_retrieval/data/judgements')
#Get judgements sorted according to their relevancy degree
judgements_sorted = get_judgements_sorted(judgements, vectors=CONFIG['vectors_file'], method=CONFIG['sentence_similarity_method'])
#df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: [round(float(e), 2) for e in x.split(' ')])
judgements_sorted['keywords'] = judgements_sorted.apply(lambda row: get_extracted_keywords(row['text']), axis=1)


