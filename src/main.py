""" This is the main module """
import warnings
from datetime import datetime

import pandas as pd
from classifiers.classifier_executor import ClassifierExecutor
from classifiers.input_data import InputData
from feature_extraction.feature_extractor import FeatureExtractor
from path_helper import get_project_root
from preprocessing.class_balancer import ClassBalancer
from preprocessing.corpus import build_corpus

# Configs
from preprocessing.data_preparation import prepare_and_merge_datasets
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None
warnings.simplefilter(action="ignore", category=FutureWarning)


def run_preprocessing(run_from_scratch):
    """ Run data preprocessing if run_from_scratch=True """
    if run_from_scratch:
        # prepare corpus
        print("\nPreparing data ...")
        prepare_and_merge_datasets()
        df_preprocessed = pd.read_csv(
            str(get_project_root()) + "/data/preprocessed/dataset.csv", index_col=0
        )
        return df_preprocessed
    else:
        df_preprocessed = pd.read_csv(
            str(get_project_root()) + "/data/preprocessed/dataset.csv", index_col=0
        )
        return df_preprocessed


def run_feature_extraction_create_corpus(run_from_scratch, df_preprocessed):
    """ Run corpus building if run_from_scratch=True """
    if run_from_scratch:
        df_corpus = build_corpus(df_preprocessed)
        df_corpus.to_csv(
            str(get_project_root()) + "/data/extracted_features/corpus.csv"
        )
        return df_corpus
    else:
        df_corpus = pd.read_csv(
            str(get_project_root()) + "/data/extracted_features/corpus.csv"
        )
        return df_corpus


def run_feature_extraction(run_from_scratch, df_corpus):
    """ Run feature extraction if run_from_scratch=True """
    if run_from_scratch:
        print("\nExtracting features ...")
        df_extracted_features = FeatureExtractor(df_corpus).get_df_with_all_features()
        df_extracted_features = df_extracted_features.drop(
            ["original_content", "content", "tokens", "pos", "stems"], axis=1
        )
        df_extracted_features.to_csv(
            str(get_project_root()) + "/data/extracted_features/extracted_features.csv"
        )
        return df_extracted_features
    else:
        df_extracted_features = pd.read_csv(
            str(get_project_root()) + "/data/extracted_features/extracted_features.csv"
        )
        return df_extracted_features


if __name__ == "__main__":
    preprocessing = False
    corpus = False
    feature_extraction = False

    df_preprocessed = run_preprocessing(preprocessing)
    df_corpus = run_feature_extraction_create_corpus(corpus, df_preprocessed)
    df_extracted_features = run_feature_extraction(feature_extraction, df_corpus)

    # unchanged dataset
    raw_text_features = df_preprocessed["content"]
    raw_text_labels = df_preprocessed["class"]
    extracted_features = df_extracted_features.loc[
        :, df_extracted_features.columns != "class"
    ]
    labels = df_extracted_features["class"]

    # do balancing, i.e. over- and undersampling
    input_data = InputData(
        raw_text_features, raw_text_labels, extracted_features, labels
    )

    # run classifiers
    print("\nRunning classifiers ...")
    classifier_executor = ClassifierExecutor(input_data.get_datasets())
    df_results = classifier_executor.get_results()
    df_results.to_csv(str(get_project_root()) + "/results/results.csv")
