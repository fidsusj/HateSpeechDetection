""" This is the main module """
import warnings
from datetime import datetime

import pandas as pd
from classifiers.classifier_executor import ClassifierExecutor
from feature_extraction.feature_extractor import FeatureExtractor
from path_helper import get_project_root
from preprocessing.class_balancer import ClassBalancer
from preprocessing.corpus import build_corpus
from sklearn.model_selection import train_test_split

# Configs
pd.options.mode.chained_assignment = None
warnings.simplefilter(action="ignore", category=FutureWarning)

if __name__ == "__main__":
    # prepare corpus
    print("\nPreparing data ...")
    # prepare_and_merge_datasets()
    df_dataset = pd.read_csv(
        str(get_project_root()) + "/data/preprocessed/dataset.csv", index_col=0
    )
    df_dataset = build_corpus(df_dataset)
    df_dataset.to_csv(str(get_project_root()) + "/data/extracted_features/corpus.csv")

    # extract features
    print("\nExtracting features ...")
    df_extracted_features = FeatureExtractor(df_dataset).get_df_with_all_features()
    df_extracted_features = df_extracted_features.drop(
        ["original_content", "content", "tokens", "pos", "stems"], axis=1
    )
    df_extracted_features.to_csv(
        str(get_project_root()) + "/data/extracted_features/extracted_features.csv"
    )

    # run classifiers
    print("\nRunning classifiers ...")
    features = df_extracted_features.loc[:, df_extracted_features.columns != "class"]
    labels = df_extracted_features["class"]

    # do balancing, i.e. over- and undersampling
    balancer = ClassBalancer(features, labels)
    # undersampled_x, undersampled_y = balancer.undersample()
    # oversampled_x, oversampled_y = balancer.oversample()
    datasets = [
        ("unchanged", (features, labels)),
        ("undersampled", balancer.undersample()),
        ("oversampled", balancer.oversample()),
    ]
    for title, (features, labels) in datasets:
        print("\n", title)
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=0.1,
            random_state=42,
        )
        print(datetime.now())
        classifier_executor = ClassifierExecutor(X_train, y_train, X_test, y_test)
        print(datetime.now())

        print("\nEvaluation results:")
        print(classifier_executor.get_evaluation_results())
