import json
import pandas as pd
import re
import string
import collections
punctuations = set(string.punctuation)
punctuations.add('• ')

def load_jsonl(input_path):
    """
    Read list of objects from a JSON lines file
    :param input_path: name of JSONL file with one dictionary/object per line
    :return: list of json
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data

def get_dataframe(infile):
    """
    Read jsonl file and return a dataframe
    :param {string} infile:
    :return: {dataframe} which contains one judgement per line
    """
    data = load_jsonl(infile)
    df = pd.DataFrame.from_records(data)
    return df

def get_processed_texts(texts):
    """
    Split sentences in list of sentences, remove punctuation in sentences
    :param {text} texts:
    :return: {list} list of sentences without punctuations inside
    """
    texts = re.split(';|\n\n|,',texts)
    return [''.join(ch for ch in x if ch not in punctuations).strip() for x in texts if x]

def remove_stop_sentence(text, stop_sentences):
    """
    Remove sentence if the sentence's occurrence frequency if high
    :param text: original text
    :param stop_sentences: stop sentences
    :return: sentences processed
    """
    text_tokenized = [t.lower() for t in text if t.lower() not in stop_sentences]
    if len(text_tokenized) == 0:
        return text
    else:
        new_text = ' '.join(text_tokenized)
    return new_text

def store_decomposed_sentences_df(infile):
    df = get_dataframe(infile)
    #df = df[:1000]
    df['judgementId'] = df.index + 1
    df['text_tokenized'] = df['text'].apply(lambda x: get_processed_texts(x))
    stop_sentences = get_stop_sentences(df['text_tokenized'])
    df['text'] = df['text_tokenized'].apply(lambda x: remove_stop_sentence(x, stop_sentences))
    output_df = df[['judgementId', 'text']].copy()
    output_df.to_csv('information_retrieval/data/judgements', header=True)

def get_stop_sentences(texts):
    texts = [text for texts in list(texts) for text in texts]
    counter=collections.Counter(texts)
    stop_sents = [sent.lower() for (sent, occ) in counter.most_common(850) if len(sent) > 2 and '€' not in sent and len(sent.split()) > 1 and len(sent.split()) < 5]
    return stop_sents

