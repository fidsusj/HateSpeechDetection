""" This is the main module """
import warnings

import pandas as pd
from classifiers.classifier_executor import ClassifierExecutor
from classifiers.neural_network_executor import NeuralNetworkExecutor
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

    # run classifiers
    print("\nRunning classifiers ...")
    raw_text_features = df_preprocessed["content"]
    raw_text_labels = df_preprocessed["class"]
    extracted_features = df_extracted_features.loc[
        :, df_extracted_features.columns != "class"
    ]
    labels = df_extracted_features["class"]

    def run_classifiers(features, labels, executor):
        # do balancing, i.e. over- and undersampling
        balancer = ClassBalancer(features, labels)
        # undersampled_x, undersampled_y = balancer.undersample()
        try:
            oversampled_x, oversampled_y = balancer.oversample()
        except:
            # TODO: hacky, needs to be adapted
            oversampled_x, oversampled_y = features, labels
        datasets = [
            ("unchanged", (features, labels)),
            ("undersampled", balancer.undersample()),
            ("oversampled", (oversampled_x, oversampled_y)),
        ]

        results = []
        for title, (features, labels) in datasets:
            print(
                " -> Run executor: {} with dataset: {}".format(executor.__name__, title)
            )
            X_train, X_test, y_train, y_test = train_test_split(
                features,
                labels,
                test_size=0.2,
                random_state=42,
            )
            classifier_executor = executor(X_train, y_train, X_test, y_test)

            print(classifier_executor.get_evaluation_results())
            results.append(classifier_executor.get_evaluation_results())

        return results[0], results[1], results[2]

    df_unchanged0, df_undersampled0, df_oversampled0 = run_classifiers(
        extracted_features, labels, ClassifierExecutor
    )
    df_unchanged1, df_undersampled1, df_oversampled1 = run_classifiers(
        [[i] for i in raw_text_features.values.tolist()],
        raw_text_labels,
        NeuralNetworkExecutor,
    )

    df_results_unchanged = pd.concat([df_unchanged0, df_unchanged1])
    df_results_undersampled = pd.concat([df_undersampled0, df_undersampled1])
    df_results_oversampled = pd.concat([df_oversampled0, df_oversampled1])

    print("\nEvaluation results:")
    print("-> unchanged:")
    print(df_results_unchanged)
    df_results_unchanged.to_csv(str(get_project_root()) + "/results/unchanged.csv")
    print("-> undersampled:")
    print(df_results_undersampled)
    df_results_undersampled.to_csv(
        str(get_project_root()) + "/results/undersampled.csv"
    )
    print("-> oversampled:")
    print(df_results_oversampled)
    df_results_oversampled.to_csv(str(get_project_root()) + "/results/oversampled.csv")
