from __future__ import absolute_import
import sys
import pandas as pd
import json
sys.path.append('/home/yichun/projects/information_retrieval')
from process import store_decomposed_sentences_df
from compute_similarity.sort import compute_similarities
from extractor_keywords.get_topic_based_on_keywords import get_extracted_keywords, get_topic

CONFIG = json.load(open("information_retrieval/config.json"))
store_decomposed_sentences_df('information_retrieval/data/judgements.jsonl') # uncomment this line the first time
judgements = pd.read_csv('information_retrieval/data/judgements')
#Get judgements sorted according to their relevancy degree
judgements = compute_similarities(judgements, vectors=CONFIG['vectors_file'], method=CONFIG['sentence_similarity_method'])
#df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: [round(float(e), 2) for e in x.split(' ')])
judgements['keywords'] = judgements.apply(lambda row: get_extracted_keywords(row['text']), axis=1)
judgements['topic_keywords'] = judgements.apply(lambda row: get_topic(row['keywords']), axis=1)
judgements = judgements.sort_values(by='diff_top1sim_top2sim', ascending=True)

writer = pd.ExcelWriter("information_retrieval/result.xlsx".format(CONFIG['sentence_similarity_method']), engine='xlsxwriter')
judgements[['judgementId', 'text', 'keywords', 'topic_keywords', 'topic_similarity','sim_a','sim_b', 'sim_c', 'sim_d',
            'sim_e', 'sim_f', 'sim_g', 'diff_top1sim_top2sim']].to_excel(writer,
            sheet_name = CONFIG['sentence_similarity_method'])
writer.save()
