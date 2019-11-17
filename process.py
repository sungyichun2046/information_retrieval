import json
import pandas as pd
import re
import string
punctuations = set(string.punctuation)

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

def store_decomposed_sentences_df(infile):
    df = get_dataframe(infile)
    df['judgementId'] = df.index
    df['text'] = df['text'].apply(lambda x: get_processed_texts(x))
    judgements = df.set_index(['judgementId'])['text'].apply(pd.Series).stack()
    judgements.columns = ['judgementId','lineId', 'text']
    # df.sort_values(by='diff_top1_top2', ascending=False)
    judgements.to_csv('data/judgements', header=True)


