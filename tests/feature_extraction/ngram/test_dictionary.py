from unittest import TestCase

import pandas as pd

from src.feature_extraction.ngram.dictionary import Dictionary


class TestDictionary(TestCase):
    raw_data_hate = {
        "class": [0, 0, 0, 0, 0],
        "content": ["i hate you", "fuck it", "fuck it", "i hate you", "i hate you"],
    }
    raw_data_neutral = {
        "class": [1, 1, 1, 1, 1],
        "content": [
            "hello world",
            "this is a cute test",
            "hello world",
            "hello world",
            "this is a cute test",
        ],
    }
    df_hate = pd.DataFrame(data=raw_data_hate, columns=["class", "content"])
    df_neutral = pd.DataFrame(data=raw_data_neutral, columns=["class", "content"])
    dictionary_size = 5
    test_dictionary = Dictionary()

    def test_extract_features(self):
        df_feature = self.test_dictionary.extract_features(self.df_hate[:1])
        raw_data_expected = {
            "class": [0],
            "content": ["i hate you"],
            "number_of_hateful_words": [0],
            "number_of_neutral_words": [0],
        }
        expected_df = pd.DataFrame(raw_data_expected)
        print(df_feature)
        print(expected_df)
        self.assertTrue(df_feature.equals(expected_df))

    def test__check_if_list_contains_words(self):
        matches = self.test_dictionary._check_if_list_contains_words(
            "hello my name is word list hello", {"hello", "my"}
        )
        self.assertEqual(3, matches)

    def test__transform_df_column_to_one_list(self):
        expected_list = "i hate you fuck it fuck it i hate you i hate you"
        self.assertEqual(
            expected_list,
            self.test_dictionary._transform_df_column_to_one_list(self.df_hate),
        )

    def test__sort_df_by_column_and_return_top_n_items(self):
        df = pd.DataFrame({"name1": [1], "name2": [6]}, index=[0, 1])
        expected_result = ["name2", "name1"]
        self.assertEqual(
            expected_result,
            self.test_dictionary._sort_df_by_column_and_return_top_n_items(df, 0, 1, 2),
        )

    def test__get_distinct_list_elements(self):
        list1 = [1, 2, 3]
        list2 = [3, 4, 5]
        self.assertEqual(
            {1, 2}, self.test_dictionary._get_distinct_list_elements(list1, list2)
        )
        self.assertEqual(
            {4, 5}, self.test_dictionary._get_distinct_list_elements(list2, list1)
        )
