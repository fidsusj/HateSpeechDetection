""" Module extracts counts of hate speech specific unigrams, bigrams and trigrams """

import collections

import pandas as pd
from nltk.util import ngrams

from src.preprocessing.corpus import build_corpus


class NGram:
    """ Class to extract ngram counts for each document based on a ngram dictionary """

    def __init__(self):
        self.dictionary_unigrams = []
        self.dictionary_bigrams = []
        self.dictionary_trigrams = []

    def fit(self, df, unigram_threshold=0, bigram_threshold=0, trigram_threshold=0):
        """ Method to create the hate speech dictionaries for ngrams """
        hate_speech_instances = df[df["class"] == 0]
        corpus = hate_speech_instances["tokens"].tolist()
        flatten = lambda t: [item for sublist in t for item in sublist]
        corpus = flatten(corpus)

        unigrams = list(ngrams(corpus, 1))
        unigram_frequencies = collections.Counter(unigrams)
        self.dictionary_unigrams = [
            unigram
            for unigram in list(set(unigrams))
            if unigram_frequencies[unigram] > unigram_threshold
        ]

        bigrams = list(ngrams(corpus, 2))
        bigram_frequencies = collections.Counter(bigrams)
        self.dictionary_bigrams = [
            bigram
            for bigram in list(set(bigrams))
            if bigram_frequencies[bigram] > bigram_threshold
        ]

        trigrams = list(ngrams(corpus, 3))
        trigram_frequencies = collections.Counter(trigrams)
        self.dictionary_trigrams = [
            trigram
            for trigram in list(set(trigrams))
            if trigram_frequencies[trigram] > trigram_threshold
        ]

    def extract_features(self, df):
        """ Adds the ngram counts to each document """

        df["hate_speech_unigrams"] = df["tokens"].apply(
            lambda tokens: len(
                set(list(ngrams(tokens, 1))) & set(self.dictionary_unigrams)
            )
        )
        df["hate_speech_bigrams"] = df["tokens"].apply(
            lambda tokens: len(
                set(list(ngrams(tokens, 2))) & set(self.dictionary_bigrams)
            )
        )
        df["hate_speech_trigrams"] = df["tokens"].apply(
            lambda tokens: len(
                set(list(ngrams(tokens, 3))) & set(self.dictionary_trigrams)
            )
        )


if __name__ == "__main__":
    df_dataset = pd.read_csv("../../data/preprocessed/dataset.csv", index_col=0)
    df_dataset = build_corpus(df_dataset)

    ngram_extractor = NGram()
    ngram_extractor.fit(
        df_dataset, unigram_threshold=10, bigram_threshold=8, trigram_threshold=2
    )
    ngram_extractor.extract_features(df_dataset)

    print(df_dataset["hate_speech_bigrams"])
