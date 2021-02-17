from unittest import TestCase

import pandas as pd
from feature_extraction.feature_extractor import FeatureExtractor
from feature_extraction.semantic.words import Words


class TestFeatureExtractor(TestCase):
    def test_get_df_with_all_features(self):
        data = ["Hello", "Hello2", "Hello3", "Hello4"]
        df = pd.DataFrame(data=data, columns=["text"])
        feature_extractor = object.__new__(FeatureExtractor)
        feature_extractor.df_with_all_extracted_features = df
        self.assertTrue(feature_extractor.get_df_with_all_features().equals(df))

    def test__extract_all_features(self):
        feature_extractor = object.__new__(FeatureExtractor)
        data = [
            ["Hello", "hello1"],
            ["Hello2", "hello2"],
            ["Hello3", "hello3"],
            ["Hello4", "hello4"],
        ]
        df = pd.DataFrame(data=data, columns=["original_content", "content"])
        data_real = [
            ["Hello", "hello1", 0, 0, 0, 1, 0],
            ["Hello2", "hello2", 0, 0, 0, 1, 0],
            ["Hello3", "hello3", 0, 0, 0, 1, 0],
            ["Hello4", "hello4", 0, 0, 0, 1, 0],
        ]
        real = pd.DataFrame(
            data=data_real,
            columns=[
                "original_content",
                "content",
                "number_of_interjections",
                "number_of_all_caps_words",
                "number_of_quotation_marks",
                "number_of_words",
                "number_of_laughing_expressions",
            ],
        )
        extracted = feature_extractor._extract_all_features(df, [Words])
        self.assertTrue(extracted.equals(real))

    def test__constructor(self):
        data = [
            [0, "Hello", "hello1", "hel", "POS", ["tok", "en"]],
            [0, "Hello2", "hello2", "hel", "POS", ["tok", "en"]],
            [1, "Hello3", "hello3", "hel", "POS", ["tok", "en"]],
            [1, "Hello4", "hello4", "hel", "POS", ["tok", "en"]],
        ]
        df = pd.DataFrame(
            data=data,
            columns=["class", "original_content", "content", "stems", "pos", "tokens"],
        )
        feature_extractor = FeatureExtractor(df)
        df_extracted = feature_extractor.get_df_with_all_features()
        expected = [
            "class",
            "original_content",
            "content",
            "stems",
            "pos",
            "tokens",
            "number_of_exclamation_mark",
            "number_of_question_mark",
            "number_of_full_stop_mark",
            "number_of_hateful_words",
            "number_of_neutral_words",
            "number_of_interjections",
            "number_of_all_caps_words",
            "number_of_quotation_marks",
            "number_of_words",
            "number_of_laughing_expressions",
            "hate_speech_unigrams",
            "hate_speech_bigrams",
            "hate_speech_trigrams",
            "pattern_count",
            "sentiment",
            "topic",
        ]
        self.assertTrue(df_extracted.columns.values.tolist() == expected)
