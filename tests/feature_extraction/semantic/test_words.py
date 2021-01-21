from unittest import TestCase

import pandas as pd

from src.feature_extraction.semantic.words import Words


class TestWords(TestCase):
    raw_data = {
        "class": [0, 1, 0, 1],
        "content": [
            "oh cool",
            "oh caps",
            """what a "quote" and ''test`` """,
            "laughter: haha lol lmao",
        ],
        "original_content": [
            "oh cool",
            "OH CAPS",
            """what a "quote" and ''test`` """,
            "laughter: haha lol lmao",
        ],
    }

    def test_extract_features(self):
        df = pd.DataFrame(data=self.raw_data)
        expected_data = {
            "class": [0, 1, 0, 1],
            "content": [
                "oh cool",
                "oh caps",
                """what a "quote" and ''test`` """,
                "laughter: haha lol lmao",
            ],
            "original_content": [
                "oh cool",
                "OH CAPS",
                """what a "quote" and ''test`` """,
                "laughter: haha lol lmao",
            ],
            "number_of_interjections": [1, 1, 0, 0],
            "number_of_all_caps_words": [0, 2, 0, 0],
            "number_of_quotation_marks": [0, 0, 4, 0],
            "number_of_words": [2, 2, 5, 4],
            "number_of_laughing_expressions": [0, 0, 0, 3],
        }
        expected_df = pd.DataFrame(data=expected_data)
        new_df = Words().extract_features(df)
        self.assertTrue(new_df.equals(expected_df))
