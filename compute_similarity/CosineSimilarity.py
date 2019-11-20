import numpy as np
from compute_similarity.CIV_ComputeSimilarity import CIV_ComputeSimilarity
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial import distance

class CosineSimilarity(CIV_ComputeSimilarity):

    def get_similarity_between_two_sentences(self, tokens1, tokens2, vectorizer=None):
        if vectorizer:
            return self.get_cosine_distance_wordembedding_method(tokens1, tokens2, vectorizer)

    def get_cosine_distance_wordembedding_method(self, tokens1, tokens2, vectorizer):
        vector_1 = np.mean([vectorizer[word] for word in tokens1],axis=0)
        vector_2 = np.mean([vectorizer[word] for word in tokens2],axis=0)
        similarity = 1 - distance.cosine(vector_1, vector_2)
        return similarity

    def get_cosine_distance_countvectorizer_method(self, s1, s2):
        """
        :param {string} s1:
        :param {string} s2:
        :return: similarity
        """

        allsentences = [s1 , s2]
        vectorizer = CountVectorizer()
        all_sentences_to_vector = vectorizer.fit_transform(allsentences)
        text_to_vector_v1 = all_sentences_to_vector.toarray()[0].tolist()
        text_to_vector_v2 = all_sentences_to_vector.toarray()[1].tolist()
        # distance of similarity
        cosine = distance.cosine(text_to_vector_v1, text_to_vector_v2)
        similairty = round((1-cosine)*100,2)
        return similairty
