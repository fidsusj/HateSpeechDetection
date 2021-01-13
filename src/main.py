""" This is the main module """

import pandas as pd
from feature_extraction.sentiment.vader import get_sentiment_features
from preprocessing.corpus import build_corpus

# Configs
pd.options.mode.chained_assignment = None

if __name__ == "__main__":
    # prepare_and_merge_datasets()
    df_dataset = pd.read_csv("./src/data/preprocessed/dataset.csv", index_col=0)
    df_dataset = build_corpus(df_dataset)
    df_dataset = get_sentiment_features(df_dataset)
    print(df_dataset)
