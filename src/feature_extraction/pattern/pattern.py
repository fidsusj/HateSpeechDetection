""" Module extracts count of typical hate speech syntactic patterns """

from collections import Counter

import pandas as pd
from preprocessing.corpus import build_corpus


class Pattern:
    """ Class to extract the hate speech pattern count """

    def __init__(self, min_pattern_size=2, max_pattern_size=2, threshold=500):
        self.patterns = []
        self.min_pattern_size = min_pattern_size
        self.max_window_size = max_pattern_size
        self.threshold = threshold

    def fit(self, df):
        """ Method to create the hate speech pattern list """

        hate_speech_documents = df[df["class"] == 0]
        for window_size in range(self.min_pattern_size, self.max_window_size + 1):
            hate_speech_documents["pos"].apply(
                lambda pos: self.patterns.append(
                    [
                        tuple(pos[i : i + self.max_window_size])
                        for i in range(len(pos) - window_size + 1)
                    ]
                )
            )
        flatten = lambda t: [item for sublist in t for item in sublist]
        self.patterns = flatten(self.patterns)

        pattern_frequencies = Counter(self.patterns)
        self.patterns = [
            pattern
            for pattern in set(self.patterns)
            if pattern_frequencies[pattern] >= self.threshold
        ]

    def extract_features(self, df):
        """ Method to create the hate speech pattern count feature """

        self.fit(df)

        for window_size in range(self.min_pattern_size, self.max_window_size + 1):
            df["pattern_count"] = df["pos"].apply(
                lambda pos: len(
                    {
                        tuple(pos[i : i + self.max_window_size])
                        for i in range(len(pos) - window_size + 1)
                    }
                    & set(self.patterns)
                )
            )
        return df


if __name__ == "__main__":
    df_dataset = pd.read_csv("../../data/preprocessed/dataset.csv", index_col=0)
    df_dataset = build_corpus(df_dataset)
    pattern_extractor = Pattern()
    pattern_extractor.extract_features(df_dataset)

    print(df_dataset["pattern_count"])
