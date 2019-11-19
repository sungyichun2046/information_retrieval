#copy of https://github.com/stephenhky/PyWMD/blob/master/WordMoverDistanceDemo.ipynb
from itertools import product
from collections import defaultdict
import fasttext
from scipy.spatial.distance import euclidean
import pulp

singleindexing = lambda m, i, j: m*i+j
unpackindexing = lambda m, k: (k/m, k % m)

def tokens_to_fracdict(tokens):
    cntdict = defaultdict(lambda : 0)
    for token in tokens:
        cntdict[token] += 1
    totalcnt = sum(cntdict.values())
    return {token: float(cnt)/totalcnt for token, cnt in cntdict.items()}


# use PuLP
def word_mover_distance_probspec(first_sent_tokens, second_sent_tokens, wvmodel, lpFile=None):
    all_tokens = list(set(first_sent_tokens+second_sent_tokens))
    wordvecs = {token: wvmodel[token] for token in all_tokens}

    first_sent_buckets = tokens_to_fracdict(first_sent_tokens)
    second_sent_buckets = tokens_to_fracdict(second_sent_tokens)

    T = pulp.LpVariable.dicts('T_matrix', list(product(all_tokens, all_tokens)), lowBound=0)

    prob = pulp.LpProblem('WMD', sense=pulp.LpMinimize)
    prob += pulp.lpSum([T[token1, token2]*euclidean(wordvecs[token1], wordvecs[token2])
                        for token1, token2 in product(all_tokens, all_tokens)])
    for token2 in second_sent_buckets:
        prob += pulp.lpSum([T[token1, token2] for token1 in first_sent_buckets])==second_sent_buckets[token2]
    for token1 in first_sent_buckets:
        prob += pulp.lpSum([T[token1, token2] for token2 in second_sent_buckets])==first_sent_buckets[token1]

    if lpFile!=None:
        prob.writeLP(lpFile)
    prob.solve()
    return prob

def word_mover_distance(first_sent_tokens, second_sent_tokens, wvmodel, lpFile=None):
    prob = word_mover_distance_probspec(first_sent_tokens, second_sent_tokens, wvmodel, lpFile=lpFile)
    return pulp.value(prob.objective)

"""
USAGE
topics = {'A': 'Rupture abusive de la relation de travail',
          'B': 'Rupture abusive du contrat de travail',
          'C': 'Rupture brutale de relations commerciales établies',
          'D': 'Rupture brutale des contrats',
          'E': 'Indemnité compensatrice de rupture',
          'F': 'Indemnité compensatrice de congés payés',
          'G': 'Indemnité compensatrice de préavis'
          }
sent = "annule les fermes rappels des 4102010 23122010 et 1032011 les deux avertissements des 93 et 1152012 et la mise à pied du 8112012 • dit le licenciement sans cause réelle et sérieuse • condamne la sas transports gautier normandie à verser à xxxx • 2 849 60 € dindemnité de licenciement • 3 562 € bruts dindemnité compensatrice de préavis outre 356 20 € bruts au titre des congés payés afférents avec intérêts au taux légal à compter du 6122013 • 1 500 € de dommages et intérêts en réparation des sanctions injustifiées • 15 000 € de dommages et intérêts pour licenciement sans cause réelle et sérieuse avec intérêts au taux légal à compter de la notification de la présente décision • dit que la sas transports gautier normandie devra remettre à xxxx dans le délai dun mois à compter de la notification de la présente décision un certificat de travail et une attestation pôle emploi rectifiés ainsi quun bulletin de paie rectificatif • déboute xxxx du surplus de ses demandes • condamne la sas transports gautier normandie aux entiers dépens de première instance dappel • condamne la sas transports gautier normandie à verser à xxxx 2 000 € en application de larticle 700 du code de procédure civile"
topicId2text = {k : v.split() for k, v in topics.items()}
print(topicId2text['A'])
tokens2 = sent.split()
vectorizer = fasttext.load_model("../word_embedding/fasttext_model.bin")
for id in topicId2text:
    tokens1 = topicId2text[id]
    print("sim=", word_mover_distance_probspec(tokens1, tokens2, vectorizer))
"""
