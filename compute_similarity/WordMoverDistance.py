#copy of https://github.com/stephenhky/PyWMD/blob/master/WordMoverDistanceDemo.ipynb
from itertools import product
from collections import defaultdict
from scipy.spatial.distance import euclidean
import pulp
from compute_similarity.CIV_ComputeSimilarity import CIV_ComputeSimilarity

singleindexing = lambda m, i, j: m*i+j
unpackindexing = lambda m, k: (k/m, k % m)

class WordMoverDistance(CIV_ComputeSimilarity):
    def tokens_to_fracdict(self, tokens):
        cntdict = defaultdict(lambda : 0)
        for token in tokens:
            cntdict[token] += 1
        totalcnt = sum(cntdict.values())
        return {token: float(cnt)/totalcnt for token, cnt in cntdict.items()}

    # use PuLP
    def word_mover_distance_probspec(self, first_sent_tokens, second_sent_tokens, wvmodel, lpFile=None):
        all_tokens = list(set(first_sent_tokens+second_sent_tokens))
        wordvecs = {token: wvmodel[token] for token in all_tokens}

        first_sent_buckets = self.tokens_to_fracdict(first_sent_tokens)
        second_sent_buckets = self.tokens_to_fracdict(second_sent_tokens)

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

    def get_similarity_between_two_sentences(self, tokens1, tokens2, vectorizer, lpFile=None):
        prob = self.word_mover_distance_probspec(tokens1, tokens2, vectorizer, lpFile=lpFile)
        return pulp.value(prob.objective)

