from __future__ import absolute_import
from compute_similarity.compute import compute_similarity_with_topics, get_top_infos
from extractor_keywords.get_topic_based_on_keywords import get_extracted_keywords, get_topic
import json

vectors_file ='/home/yichun/projects/information_retrieval/word_embedding/fasttext_model_50d.bin'
sentence = 'La somme de 22.539,99 euros au titre de l’indemnité de loyers dus jusqu’à la date d’échéance du bail suite ' \
           'à la rupture du bail ; ® La somme de 1.594,81 euros au titre de la perte de budget vacances 2012 suite à ' \
           'la rupture du bail'

def evaluate():
    with (open('/home/yichun/projects/information_retrieval/data/test.json', 'r')) as f:
        corpus = json.load(f)
        for sentence, topic in corpus:
            scores = compute_similarity_with_topics(sentence, 'cosine', vectors_file)
            print('Sentece = {}\nTopic = {}\nCosine Similarities = {}, Topic identified by similarities = {}\n'.format(sentence,topic, scores, get_top_infos(scores, 'cosine')[0]))
            scores = compute_similarity_with_topics(sentence, 'wmd', vectors_file)
            print("Word Mover's Distances = {}, Topic identified by similarities = {}".format(scores, get_top_infos(scores, 'wmd')[0]))
            print("Keywords = {}\nTopic indentified by keywords= {}".format(get_extracted_keywords(sentence), get_topic(get_extracted_keywords(sentence))))
