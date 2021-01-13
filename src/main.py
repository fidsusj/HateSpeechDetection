""" This is the main module """

import pandas as pd
from feature_extraction.feature_extractor import FeatureExtractor
from gensim.models import Word2Vec
from preprocessing.corpus import build_corpus
from preprocessing.data_preparation import prepare_and_merge_datasets
from sklearn.model_selection import train_test_split

# Configs
pd.options.mode.chained_assignment = None

if __name__ == "__main__":
    prepare_and_merge_datasets()
    df_dataset = pd.read_csv("data/preprocessed/dataset.csv", index_col=0)
    df_dataset = build_corpus(df_dataset)

    # extract features
    df_extracted_features = FeatureExtractor(df_dataset).get_df_with_all_features()

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        df_extracted_features.loc[:, df_extracted_features.columns != "class"],
        df_extracted_features["class"],
        test_size=0.1,
        random_state=42,
    )
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    # run classifiers

    w2v_model = Word2Vec.load("./models/model.pickle")
    print(w2v_model.wv.most_similar("niggas"))
