from __future__ import absolute_import
import sys
sys.path.append('/home/yichun/projects/information_retrieval')
import json
import pandas as pd

topic2keywords = { 'A': [['rupture', 'relation de travail'], ['licenciement']],
                   'B': [['rupture', 'contrat de travail'], ['licenciement']],
                   'C': [['rupture', 'relations commerciales']],
                    'D': [['rupture', 'contrat'], ['rupture', 'bail'], ['licenciement']],
                    'E': [['indemnité', 'rupture'], ['indemnité', 'licenciement'], ['indemnité', 'divorce'], ['prestation', 'divorce']],
                    'F': [['indemnité', 'congés payés']],
                    'G': [['indemnité', 'préavis']]
                   }
with open('/home/yichun/projects/information_retrieval/data/keywords.json') as f:
        keywords = json.load(f)
with open('/home/yichun/projects/information_retrieval/data/stopwords_fr.json') as f:
        stopwords_fr = json.load(f)

def get_extracted_keywords(judgement):
    """
    :param {string} judgement:
    :return: list of keywords
    """
    extracted_keywords = []
    for keyword in keywords:
        if keyword in judgement:
            extracted_keywords.append(keyword)
    return extracted_keywords

def get_topic(keywords):
    """
    :param {string[]} keywords: list of keywocongés payésrds in judgement
    :return: topic identified belong to keywords
    """
    df = pd.read_csv('/home/yichun/projects/information_retrieval/data/similar_words', sep=';')
    word2similar_words = df.set_index('word')['similar_word'].to_dict()
    word2similar_words = {k: v.split(',') if ',' in v else  [v] for k, v in word2similar_words.items()}
    keywords_unified = []
    for keyword in keywords:
        for k in word2similar_words:
            if keyword in word2similar_words[k]:
                keyword = k
                break
        keywords_unified.append(keyword)
    topics_identified = set()
    for topic in topic2keywords:
        for t in topic2keywords[topic]:
            if len(set(keywords_unified).intersection(t)) == len(t):
                topics_identified.add(topic)
                break
    return list(sorted(topics_identified))
"""
topic_c = "déboute la société richa de sa demande en annulation du jugement du tribunal de commerce de lyon en date du 9 avril 2014 confirme le jugement entrepris en ce qu'il a rejeté l'exception de litispendance au profit de la cour d'appel de gent concernant la demande principale en indemnisation formée par la société distrimoto international sur le fondement de l'article l442-6 i 5º du code de commerce confirme le jugement entrepris en ce qu'il a admis l'exception de litispendance concernant la demande reconventionnelle en paiement d'une somme de 68155 67 euros l'infirme en ce qu'il s'est déclaré incompétent pour en connaître et statuant à nouveau sursoit à statuer sur la demande reconventionnelle formée par la société richa en paiement d'une somme de 68153 16 euros dans l'attente que la compétence de la cour d'appel de gent saisie de l'appel du jugement du tribunal de commerce d'oudenaarde du 7 octobre 2014 soit établie confirme le jugement entrepris en ce qu'il a dit la loi française applicable et a reconnu l'existence de relations commerciales établies entre les parties pendant 13 ans au sens de l'article l442-6 i 5º du code de commerce sursoit à statuer sur la demande d'indemnisation pour rupture brutale des relations commerciales dans l'attente de l'arrêt rendu par la cour d'appel de gent saisie de l'appel du jugement du tribunal de commerce d'oudenaarde du 7 octobre 2014 réserve les autres demandes"
keywords = get_extracted_keywords(topic_c)
print(keywords)
topic = get_topic(keywords)
print(topic)
"""
