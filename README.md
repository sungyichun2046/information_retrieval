# information_retrieval and identification of topic(s)

This project is to identify topic(s) in judgements, we already know that there are 7 topics possible,
There are two possible approaches:
1) Calculate similarities between main sentence of each judgement and 7 topics and use word embedding model to do it:
    The WMD distance measures the dissimilarity between two texts, and the cosine similarity mesures the similarity
    between two texts

2) Train a word embedding model on our 10k corpus, and find keywords closed to words in 7 topic, and identify keywords in each
    judgement in order to find the topic realted. It's good to identify several topics inside

Finally we have a corpus sorted of the form:
    1. Judgement identifier, which corresponds to the line number within the JSONL file
    2. Main sentence extracted from the judgment
    3. Similarities between main sentence and 7 topics
    4. Topic identified by similarity
    5. Keywords
    6. Topic identified by keywords

The set of topics to measure the sentences against are:
    A. Rupture abusive de la relation de travail
    B. Rupture abusive du contrat de travail
    C. Rupture brutale de relations commerciales établies
    D. Rupture brutale des contrats
    E. Indemnité compensatrice de rupture
    F. Indemnité compensatrice de congés payés
    G. Indemnité compensatrice de préavis

In the table sorted, we have the top the most "problematic" cases, that is rows in which the best topic is not that distance from the second topic.
(table sorted by diffrerenc between top1 topic similarity and top2 topic similarity)

### Table of Contents: ###


| File/module                               | Description |
| -------------                             | ------------- |
| config.json                               | parameters configurations
| process.py                                | convert jsonl to pandas, find stop_sentences, remove them in judgements
| evaluation.py                             | see similarities, keywords, topic identified by keywords/by similarities for one sentence
| data/judgements                           | csv file contain judgementId,text
| word_embedding/                           | 1) Create corpus.txt as training data 2) Train fasttext and word2vec vectors
| extractor_keywords/                       | Store keywords in file and identify it in judgements
| compute_similarity/                        | Calculate sentence similarity by wmd and cosine similarity methods

### Usage: ###

    - Load jsonl file and store the pandas object converted in csv file :
      use function `store_decomposed_sentences_df` in main.py(uncomment it)
    - Train fasttext vectors and word2vec vectors:
      run `python Fasttext.py` and `Word2vec.py`
    - Set parameters `sentence_similarity_method` and `vecotrs_file` in `config.json` and run `main.py`
      go up to the directory above `information_retrieval` folder and run `python -m information_retrieval.main`
      it will store result sorted as csv file and excel file
