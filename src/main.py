""" This is the main module """
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

from src.classifiers.classifier_executor import ClassifierExecutor
from src.feature_extraction.feature_extractor import FeatureExtractor
from src.preprocessing.corpus import build_corpus

# Configs
pd.options.mode.chained_assignment = None
warnings.simplefilter(action="ignore", category=FutureWarning)


if __name__ == "__main__":

    # prepare corpus
    print("\nPreparing data ...")
    # prepare_and_merge_datasets()
    df_dataset = pd.read_csv("./src/data/preprocessed/dataset.csv", index_col=0)
    df_dataset = build_corpus(df_dataset)

    # extract features
    print("\nExtracting features ...")
    df_extracted_features = FeatureExtractor(df_dataset).get_df_with_all_features()
    df_extracted_features = df_extracted_features.drop(
        ["original_content", "content", "tokens", "pos", "stems"], axis=1
    )

    # run classifiers
    print("\nRunning classifiers ...")
    X_train, X_test, y_train, y_test = train_test_split(
        df_extracted_features.loc[:, df_extracted_features.columns != "class"],
        df_extracted_features["class"],
        test_size=0.1,
        random_state=42,
    )
    classifier_executor = ClassifierExecutor(X_train, y_train, X_test, y_test)

    print("\nEvaluation results:")
    print(classifier_executor.get_evaluation_results())

    # w2v_model = Word2Vec.load("./models/model.pickle")
    # print(w2v_model.wv.most_similar("niggas"))
