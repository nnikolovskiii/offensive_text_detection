import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm")


def load_datasets():
    df_train = pd.read_csv('data/train_en.txt', sep='\t')
    df_val = pd.read_csv('data/val_en.txt', sep='\t')
    df_test = pd.read_csv('data/test_en.txt', sep='\t')

    return df_train, df_val, df_test


def tokenize_with_spacy(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    return tokens


def create_vocab(df_train):
    vocab = {}
    for sentence in df_train["Sentence"]:
        for tkn in tokenize_with_spacy(sentence):
            if tkn not in vocab:
                vocab[tkn] = len(vocab) + 1
    return vocab


def tokenize_df(df, vocab):
    tkns = []
    for sentence in df["Sentence"]:
        s = []
        for tkn in tokenize_with_spacy(sentence):
            if tkn in vocab:
                s.append(vocab[tkn])
        tkns.append(s)
    df["tkns"] = tkns
    return df


