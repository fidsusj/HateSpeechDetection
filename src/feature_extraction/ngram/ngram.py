""" Module extracts counts of hate speech specific unigrams, bigrams and trigrams """

from collections import Counter

from nltk.util import ngrams


class NGram:
    """ Class to extract ngram counts for each document based on a ngram dictionary """

    def __init__(self):
        self.dictionary_unigrams = []
        self.dictionary_bigrams = []
        self.dictionary_trigrams = []

    def fit(self, df, unigram_threshold, bigram_threshold, trigram_threshold):
        """ Method to create the hate speech dictionaries for ngrams """
        hate_speech_instances = df[df["class"] == 0]
        corpus = hate_speech_instances["stems"].tolist()

        unigrams = []
        bigrams = []
        trigrams = []
        for tokens in corpus:
            unigrams.extend(list(ngrams(tokens, 1)))
            bigrams.extend(list(ngrams(tokens, 2)))
            trigrams.extend(list(ngrams(tokens, 3)))

        unigram_frequencies = Counter(unigrams)
        self.dictionary_unigrams = [
            unigram
            for unigram in list(set(unigrams))
            if unigram_frequencies[unigram] >= unigram_threshold
        ]

        bigram_frequencies = Counter(bigrams)
        self.dictionary_bigrams = [
            bigram
            for bigram in list(set(bigrams))
            if bigram_frequencies[bigram] >= bigram_threshold
        ]

        trigram_frequencies = Counter(trigrams)
        self.dictionary_trigrams = [
            trigram
            for trigram in list(set(trigrams))
            if trigram_frequencies[trigram] >= trigram_threshold
        ]

    def extract_features(
        self, df, unigram_threshold=10, bigram_threshold=8, trigram_threshold=2
    ):
        """ Adds the ngram counts to each document """

        self.fit(
            df,
            unigram_threshold=unigram_threshold,
            bigram_threshold=bigram_threshold,
            trigram_threshold=trigram_threshold,
        )

        df["hate_speech_unigrams"] = df["stems"].apply(
            lambda tokens: len(
                set(list(ngrams(tokens, 1))) & set(self.dictionary_unigrams)
            )
        )
        df["hate_speech_bigrams"] = df["stems"].apply(
            lambda tokens: len(
                set(list(ngrams(tokens, 2))) & set(self.dictionary_bigrams)
            )
        )
        df["hate_speech_trigrams"] = df["stems"].apply(
            lambda tokens: len(
                set(list(ngrams(tokens, 3))) & set(self.dictionary_trigrams)
            )
        )
        return df
