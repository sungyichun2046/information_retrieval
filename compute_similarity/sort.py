from __future__ import absolute_import
from compute_similarity.WordMoverDistance import WordMoverDistance
from compute_similarity.CosineSimilarity import CosineSimilarity
import fasttext
import pandas as pd
import json
import math
from datetime import datetime

with open("information_retrieval/data/stopwords_fr.json", "r") as infile:
    stopwords_fr = json.load(infile)
topics = {'A': 'Rupture abusive de la relation de travail',
          'B': 'Rupture abusive du contrat de travail',
          'C': 'Rupture brutale de relations commerciales établies',
          'D': 'Rupture brutale des contrats',
          'E': 'Indemnité compensatrice de rupture',
          'F': 'Indemnité compensatrice de congés payés',
          'G': 'Indemnité compensatrice de préavis'
          }
synonymes = ['licenciement', 'dommages-intérêts', "dommages et intérêts", "dommages", "intérêts",
             "allocation", "compensation", "dédommagement", "indemnisation", "paiement", "pécule", "prestation",
               "départ", "expulsion",  "mise à la porte", "mise au chômage", "renvoi"
             , "illégitime", "injuste", "injustifié", "bail", "brusque"]

def get_topicId2text():
    """
    :return: topicId2text where text is  lowercase and tokenized
    """
    topicId2text = {}
    for topic in topics:
        tokens = topics[topic].split()
        tokens = [t.lower() for t in tokens if t not in stopwords_fr]
        topicId2text[topic] = tokens
    return topicId2text

def get_top_infos(sims, method):
    """
    Return top1topic, diff_top1topic_top2topic
    if method is 'wmd', top1topic is the topic has the min distance
    if mthod is 'cosine', top1topic is the topic has the max similarity score
    :param {float[]} sims: list of similarities/dissimilarities
    :param {string} method: method for mesuring similarities
    :return: top1topic, diff_top1topic_top2topic
    """
    t = [k for k, v in topics.items()]
    if method == 'wmd':
        top1topic_index, top2topic_index = sorted(range(len(sims)), key=lambda i: sims[i])[:2]
    elif method == 'cosine':
        top1topic_index, top2topic_index = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:2]
    diff_top1sim_top2sim = abs(sims[top1topic_index] - sims[top2topic_index])
    top1topic = t[top1topic_index]
    return top1topic, diff_top1sim_top2sim

def compute_similarities(judgements, method="cosine", vectors=None):
    """
    :param {dataframe} judgements:
    :param {string} method: method to compute similarity bewteen sentences
    :param vectors: vectors bin file
    :return:
    """
    vectorizer = fasttext.load_model(vectors)
    topicId2text = get_topicId2text()
    if method == "wmd":
        similarity_calculator = WordMoverDistance()
    elif method == "cosine":
        similarity_calculator = CosineSimilarity()
    #store_similarities(judgements, similarity_calculator, method, vectorizer, topicId2text)#uncomment this line for storing similarities
    judgements['sims'] = create_similarities_columns(method)
    judgements[['sim_a','sim_b', 'sim_c', 'sim_d', 'sim_e', 'sim_f', 'sim_g']] = pd.DataFrame(judgements.sims.values.tolist(), index= judgements.index)
    judgements['topic_similarity'], judgements[('diff_top1sim_top2sim')] = zip(*judgements.apply(lambda row: get_top_infos(row['sims'], method), axis=1))
    #print relevant judgements
    return judgements

def create_similarities_columns(method):
    file = "/home/yichun/projects/information_retrieval/compute_similarity/results/scores_{}".format(method)
    df = pd.read_csv(file, sep='\t', header=None)
    df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: [float(e) for e in x.split(' ')])
    #df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: [round(float(e), 2) for e in x.split(' ')])
    return df.iloc[:, 1]

def store_similarities(judgements, similarity_calculater, method, vectorizer, topicId2text):
    """
    It takes sometimes lots of times for calculating similarities, i.e wmd. So we store it in files
    :param judgements:
    :param similarity_calculater:
    :param method:
    :param vectorizer:
    :param topicId2text:
    :return:
    """
    texts = judgements['text']
    start = datetime.now()
    for i in range(len(texts)):
        try:
            tokens = texts[i].split()
            tokens = [t.lower() for t in tokens if t not in stopwords_fr]
            scores = [similarity_calculater.get_similarity_between_two_sentences(tokens, topicId2text[topic], vectorizer) for topic in topicId2text]
            scores_str = ' '.join([str(s) for s in scores])
            with open("/home/yichun/projects/information_retrieval/compute_similarity/results/scores_{}".format(method), 'a') as f:
                f.write("{}\t{}\n".format(i+1, scores_str))
        except Exception:
            pass
            with open("/home/yichun/projects/information_retrieval/compute_similarity/results/scores_{}".format(method), 'a') as f:
                f.write("{}\t{}\n".format(i+1, "error"))
        fin = datetime.now()
    #print("executing time = ", fin-start)
