""" This is the main module """
import warnings

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


def run_preprocessing(run_from_scratch):
    """ Run data preprocessing if run_from_scratch=True """
    if run_from_scratch:
        # prepare corpus
        print("\nPreparing data ...")
        # prepare_and_merge_datasets()
        df = pd.read_csv(
            str(get_project_root()) + "/data/preprocessed/dataset.csv", index_col=0
        )
        df = build_corpus(df)
        df.to_csv(str(get_project_root()) + "/data/extracted_features/corpus.csv")
        return df
    else:
        df = pd.read_csv(
            str(get_project_root()) + "/data/extracted_features/corpus.csv"
        )
        return df


def run_feature_extraction(run_from_scratch):
    """ Run feature extraction if run_from_scratch=True """
    if run_from_scratch:
        # extract features
        print("\nExtracting features ...")
        df = FeatureExtractor(df_dataset).get_df_with_all_features()
        df = df.drop(["original_content", "content", "tokens", "pos", "stems"], axis=1)
        df.to_csv(
            str(get_project_root()) + "/data/extracted_features/extracted_features.csv"
        )
        return df
    else:
        df = pd.read_csv(
            str(get_project_root()) + "/data/extracted_features/extracted_features.csv"
        )
        return df


if __name__ == "__main__":
    preprocessing = False
    feature_extraction = False

    df_dataset = run_preprocessing(preprocessing)
    df_extracted_features = run_feature_extraction(feature_extraction)

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
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=0.2,
            random_state=42,
        )
        print(len(y_train))
        print(len(y_test))

        classifier_executor = ClassifierExecutor(X_train, y_train, X_test, y_test)

        print("\nEvaluation results: {}".format(title))
        print(classifier_executor.get_evaluation_results())
