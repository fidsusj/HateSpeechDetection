from unittest import TestCase

import pandas as pd
from feature_extraction.pattern.pattern import Pattern
from preprocessing.corpus import build_corpus


class Test_Pattern(TestCase):
    raw_data = {
        "class": [0, 0, 1, 0],
        "content": [
            "John hates bitches",
            "John hates hookers",
            "John loves turtles",
            "Bitch ass nigga",
        ],
    }

    def test_pattern_count(self):
        df = build_corpus(pd.DataFrame(data=self.raw_data))

        pattern_extractor = Pattern(min_pattern_size=2, max_pattern_size=2, threshold=2)
        pattern_extractor.extract_features(df)

        self.assertEqual([2, 2, 2, 1], df["pattern_count"].tolist())
