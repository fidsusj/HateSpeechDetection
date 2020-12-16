""" This is the main module """

import pandas as pd
from preprocessing.corpus import build_corpus

# Configs
pd.options.mode.chained_assignment = None

if __name__ == "__main__":
    df_dataset = pd.read_csv("data/preprocessed/dataset.csv", index_col=0)
    df_hate_speech = build_corpus(df_dataset[df_dataset["class"] == 0])
    df_neutral_speech = build_corpus(df_dataset[df_dataset["class"] == 1])
