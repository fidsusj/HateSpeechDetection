from unittest import TestCase

import pandas as pd
from src.feature_extraction.semantic.words import Words


class TestWords(TestCase):
    raw_data = {
        "class": [0, 1],
        "content": ["oh cool", "oh caps"],
        "original_content": ["oh cool", "OH CAPS"],
    }

    def test_extract_features(self):
        df = pd.DataFrame(data=self.raw_data)
        expected_data = {
            "class": [0, 1],
            "content": ["oh cool", "oh caps"],
            "original_content": ["oh cool", "OH CAPS"],
            "number_of_interjections": [1, 1],
            "number_of_all_caps_words": [0, 2],
        }
        expected_df = pd.DataFrame(data=expected_data)
        new_df = Words().extract_features(df)
        self.assertTrue(new_df.equals(expected_df))
