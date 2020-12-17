from unittest import TestCase

import pandas as pd
from feature_extraction.semantic.special_characters import SpecialCharacters


class TestSpecialCharacters(TestCase):
    raw_data = {
        "class": [0, 1],
        "content": ["First data instance!?", "Second data instance."],
    }

    def test_extract_features(self):
        df = pd.DataFrame(data=self.raw_data)
        expected_data = {
            "class": [0, 1],
            "content": ["First data instance!?", "Second data instance."],
            "number_of_exclamation_mark": [1, 0],
            "number_of_question_mark": [1, 0],
            "number_of_full_stop_mark": [0, 1],
        }
        expected_df = pd.DataFrame(data=expected_data)
        new_df = SpecialCharacters().extract_features(df)
        self.assertTrue(new_df.equals(expected_df))

    def test__count_number_of_special_characters(self):
        df = pd.DataFrame(data=self.raw_data)
        expected_data = {
            "class": [0, 1],
            "content": ["First data instance!?", "Second data instance."],
            "number_of_exclamation_mark": [1, 0],
        }
        expected_df = pd.DataFrame(data=expected_data)
        new_df = SpecialCharacters()._count_number_of_special_characters(
            "exclamation_mark", "!", df
        )
        self.assertTrue(new_df.equals(expected_df))

    def test__count_character(self):
        sentence = "Hello ??? this is a test?!!!"
        self.assertEqual(SpecialCharacters()._count_character(sentence, "!"), 3)
        self.assertEqual(SpecialCharacters()._count_character(sentence, "?"), 4)
        self.assertEqual(SpecialCharacters()._count_character(sentence, "."), 0)
