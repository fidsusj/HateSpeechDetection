from unittest import TestCase

import pandas as pd

from src.feature_extraction.ngram.ngram import NGram
from src.preprocessing.corpus import build_corpus


class Test_Ngram(TestCase):
    raw_data = {
        "class": [0, 0, 1, 0],
        "content": [
            "John hates bitches",
            "Hating bitches every day every night",
            "John loves turtles",
            "Bitch ass nigga",
        ],
    }

    def test_unigram_count(self):
        df = build_corpus(pd.DataFrame(data=self.raw_data))

        ngram_extractor = NGram()
        ngram_extractor.extract_features(df, unigram_threshold=2)

        self.assertEqual([2, 2, 0, 1], df["hate_speech_unigrams"].tolist())

    def test_bigram_count(self):
        df = build_corpus(pd.DataFrame(data=self.raw_data))

        ngram_extractor = NGram()
        ngram_extractor.extract_features(df, bigram_threshold=2)

        self.assertEqual([1, 1, 0, 0], df["hate_speech_bigrams"].tolist())

    def test_trigram_count(self):
        df = build_corpus(pd.DataFrame(data=self.raw_data))

        ngram_extractor = NGram()
        ngram_extractor.extract_features(df, trigram_threshold=2)

        self.assertEqual([0, 0, 0, 0], df["hate_speech_trigrams"].tolist())
