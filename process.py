import json
import pandas as pd
import re
import collections
with open("/home/yichun/projects/information_retrieval/data/punctuation.json", "r") as infile:
    punctuations = json.load(infile)

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
    :param text: original list of sentences
    :param stop_sentences: stop sentences
    :return: sentences processed
    """
    list_sentences_without_stop_sent = [sent.lower() for sent in text if sent.lower() not in stop_sentences]
    words = [w for sent in list_sentences_without_stop_sent for w in sent.split(' ')]
    if len(words) < 15 :
        return ' '.join([get_sentenc_without_punctuation(sent).lower() for sent in text])
    else:
        new_text = ' '.join([get_sentenc_without_punctuation(sent) for sent in list_sentences_without_stop_sent])
    return new_text

def get_sentenc_without_punctuation(sent):
    for p in punctuations:
        if p in sent:
            sent = sent.replace(p, '')
    return sent

def store_decomposed_sentences_df(infile):
    df = get_dataframe(infile)
    df['judgementId'] = df.index + 1
    df['list_of_sents'] = df['text'].apply(lambda x: get_processed_texts(x))
    stop_sentences = get_stop_sentences(df['list_of_sents'])
    df['text'] = df['list_of_sents'].apply(lambda x: remove_stop_sentence(x, stop_sentences))
    output_df = df[['judgementId', 'text']].copy()
    output_df.to_csv('/home/yichun/projects/information_retrieval/data/judgements', header=True, index=False)

def get_stop_sentences(texts):
    texts = [text for texts in list(texts) for text in texts]
    counter=collections.Counter(texts)
    stop_sents = [sent.lower() for (sent, occ) in counter.most_common(1550) if len(sent) > 2 and 'euros' not in sent.lower() and '€' not in sent and len(sent.split()) > 1 and len(sent.split()) < 5]
    with open("/home/yichun/projects/information_retrieval/data/stop_sentences.json", "w") as f:
        json.dump(stop_sents, f, indent=1, sort_keys=True)
    return stop_sents

def tokenizer(text):
    """
    :param text: string
    :return: list of token
    """
    # split text with multiple delimiters

    if str(text).isdigit() or type(text) == int:
        tokens = str(text)
    else:
        tokens = re.split(
                r'\s+',
                re.sub(r"[,\!?'’/\(\)\.]", " ", text).strip()
        )
    return tokens

