import json

synonymes = ['licenciement', 'dommages-intérêts', "dommages et intérêts",
             "allocation", "compensation", "dédommagement", "indemnisation", "paiement", "pécule", "prestation",
               "départ", "expulsion",  "mise à la porte", "mise au chômage", "renvoi"
             , "illégitime", "injuste", "injustifié", "bail", "brusque"]
topics = {'A': 'Rupture abusive de la relation de travail',
          'B': 'Rupture abusive du contrat de travail',
          'C': 'Rupture brutale de relations commerciales établies',
          'D': 'Rupture brutale des contrats',
          'E': 'Indemnité compensatrice de rupture',
          'F': 'Indemnité compensatrice de congés payés',
          'G': 'Indemnité compensatrice de préavis'
          }
with open("information_retrieval/data/stopwords_fr.json", "r") as infile:
    stopwords_fr = json.load(infile)

def get_keywords():
    texts = [v.split(' ') for k, v in topics.items()]
    keywords = list(set([w for text in texts for w in text if w not in stopwords_fr]))
    keywords.extend(synonymes)
    return keywords

def get_extracted_keywords(judgement):
    keywords = get_keywords()
    tokens = judgement.split()
    judgement = [w for w in tokens if w not in stopwords_fr]
    judgement = ' '.join(ch for ch in judgement)
    extracted_keywords = []
    for keyword in keywords:
        if keyword in judgement:
            if keyword in judgement:
                extracted_keywords.append(keyword)
    return extracted_keywords
