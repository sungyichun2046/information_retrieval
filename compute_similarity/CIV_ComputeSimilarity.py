from abc import ABC, abstractmethod

class CIV_ComputeSimilarity(ABC):

    @abstractmethod
    def get_similarity_between_two_sentences(self):
        """
        Measures similarity(i.e cosine similarity) or dissimilarity/distance(i.e WMD distance) between two texts
        :return:
        """
        pass
