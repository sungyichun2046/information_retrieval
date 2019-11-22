from __future__ import absolute_import
import sys
sys.path.append('/home/yichun/projects/information_retrieval')
import pandas as pd
from process import tokenizer
from gensim.models import Word2Vec

class Word2vec():
    def __init__(self, infile=None):
        self.model =None
        if infile:
            self.load(infile)

    def train(self, sentences, outfile):
        """
        :param sentences: list of tokens
        :param outfile:
        :return:
        """
        self.model = Word2Vec(sentences=list(sentences), size=100, window=5, min_count=5, workers=4, sg=0)
        self.model.save(outfile)

    def load(self, infile):
        self.model = Word2Vec.load(infile)

    def retrain(self, new_sentences):
        """
        Continue to train
        """
        self.model.train(new_sentences, total_examples=len(new_sentences), epochs=1)

    def get_word_similarity(self, word1, word2):
        return self.model.similarity(word1, word2)

    def get_most_similar_words(self, word=None):
        similar_words = self.model.wv.most_similar(word)
        #return [w for sim, w in similarity_words]
        return similar_words

"""
#USAGE
infile = '/home/yichun/projects/information_retrieval/word_embedding/word2vec_model_100d'
vectorizer = Word2vec()
corpus = pd.read_csv('/home/yichun/projects/information_retrieval/data/judgements')
corpus['text_tokenized'] = corpus['text'].apply(lambda x: tokenizer(x))
vectorizer.train(list(corpus['text_tokenized']), infile)
similar_words = vectorizer.get_most_similar_words(word='indemnité')
sim = vectorizer.get_word_similarity('arrêt', 'rupture')
print(similar_words)
print(sim)
"""





