""" Module runs all classifiers in this directory and returns a dataframe with performance metrices """
import multiprocessing
from datetime import datetime
from multiprocessing import Pool

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from classifiers.hyperparameters import hyperparameter_search_space
from classifiers.lstm import LSTMClassifier
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class ClassifierExecutor:
    """ Class runs all classifiers """

    def __init__(self, datasets):
        classical_ml_methods = [
            ["random_forest", RandomForestClassifier()],
            ["decision_tree", DecisionTreeClassifier()],
            ["svm", SVC()],
            ["logistic_regression", LogisticRegression()],
        ]
        neural_network_methods = [["lstm", LSTMClassifier()]]

        classical_ml_methods_run_params = self._create_run_parameters(
            classical_ml_methods, ["unchanged", "undersampled", "oversampled"], datasets
        )
        neural_network_methods_run_params = self._create_run_parameters(
            neural_network_methods, ["unchanged", "undersampled"], datasets
        )

        start_time = datetime.now()
        with Pool(multiprocessing.cpu_count()) as p:
            neural_network_methods_results = pd.concat(
                p.starmap(self.run_nn, neural_network_methods_run_params)
            )
            classical_ml_methods_results = pd.concat(
                p.starmap(self.run_classical_ml, classical_ml_methods_run_params)
            )

        self.results = pd.concat(
            [neural_network_methods_results, classical_ml_methods_results]
        )
        print(
            "Started computation: {}; Ended computation: {}".format(
                start_time, datetime.now()
            )
        )
        print("Results: \n{}".format(self.results))

    def _create_run_parameters(self, classifier_list, dataset_type_list, datasets):
        run_parameters = []
        for dataset_type in dataset_type_list:
            for classifier in classifier_list:
                run_parameters.append((classifier, dataset_type, datasets))
        return run_parameters

    def get_results(self):
        """ Getter for the results """
        return self.results

    def run_classical_ml(self, classifier, dataset_type, datasets):
        """Runs passed classifier (classical ML methods) with passed dataset
        Parameters:
            classifiers: list containing [classifier_name, classifier_class]
            dataset_type: string ("raw_datasets"|"extracted_datasets")
            datasets: dict with datasets (as in InputData)
        Return:
            df_evaluation_results: dataframe with the evaluation results
        """
        X_train, y_train, X_test, y_test = self._extract_train_and_test(
            datasets, "extracted_datasets", dataset_type
        )
        classifier_name = classifier[0]
        classifier_class = classifier[1]

        print("OPTIMIZE {}".format(classifier_name))
        classifier_pipeline = Pipeline([("classifier", classifier_class)])
        gridsearch = RandomizedSearchCV(
            classifier_pipeline,
            hyperparameter_search_space[classifier_name],
            cv=5,
            verbose=0,
            n_jobs=-1,
        )
        gridsearch.fit(X_train, y_train)
        best_model = gridsearch.best_estimator_
        classifier_evaluation = self._evaluate_classifier_on_test_set(
            best_model, dataset_type, X_test, y_test, classifier_name
        )

        df_evaluation_results = pd.DataFrame(
            data=[classifier_evaluation],
            columns=["classifier", "dataset", "precision", "recall", "accuracy", "f1"],
        )
        return df_evaluation_results

    def run_nn(self, classifier, dataset_type, datasets):
        """Runs passed classifier (nn approach) with passed dataset
        Parameters:
        Parameters:
            classifiers: list containing [classifier_name, classifier_class]
            dataset_type: string ("raw_datasets"|"extracted_datasets")
            datasets: dict with datasets (as in InputData)
        Return:
            df_evaluation_results: dataframe with the evaluation results
        """
        X_train, y_train, X_test, y_test = self._extract_train_and_test(
            datasets, "raw_datasets", dataset_type
        )

        classifier_name = classifier[0]
        classifier_class = classifier[1]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=42,
        )
        classifier_class.fit(X_train, y_train, X_val, y_val)
        classifier_evaluation = self._evaluate_classifier_on_test_set(
            classifier_class, dataset_type, X_test, y_test, classifier_name
        )
        df_evaluation_results = pd.DataFrame(
            data=[classifier_evaluation],
            columns=["classifier", "dataset", "precision", "recall", "accuracy", "f1"],
        )
        return df_evaluation_results

    def _extract_train_and_test(self, datasets, which_dataset, dataset_type):
        X_train = datasets[which_dataset][dataset_type]["X_train"]
        y_train = datasets[which_dataset][dataset_type]["y_train"]
        X_test = datasets[which_dataset][dataset_type]["X_test"]
        y_test = datasets[which_dataset][dataset_type]["y_test"]
        return X_train, y_train, X_test, y_test

    def _evaluate_classifier_on_test_set(
        self, best_model, dataset_type, X_test, y_test, classifier_name
    ):
        """Evaluates model on test set
        Parameters:
            best_model: model to be tested
            X_test: test dataset features
            y_test: test dataset labels
            classifier_name: string of the classifier name
        Return:
            array of [classifier_name, precision, recall, accuracy, f1]
        """
        y_predicted = best_model.predict(X_test)
        precision, recall, accuracy, f1 = self._calculate_performance_metrices(
            y_test, y_predicted, classifier_name
        )
        return [classifier_name, dataset_type, precision, recall, accuracy, f1]

    def _calculate_performance_metrices(self, y, y_hat, classifier_name):
        """Calculates performance metrices of the model
        Parameters:
            y: test dataset labels
            y_hat: preidcted labels
            classifier_name: string of the classifier name
        Return:
            precision, recall, accuracy, f1
        """
        cm = confusion_matrix(y, y_hat)
        precision = precision_score(y, y_hat)
        recall = recall_score(y, y_hat)
        accuracy = accuracy_score(y, y_hat)
        f1 = f1_score(y, y_hat)
        plt.figure()
        sns.heatmap(cm, cmap="PuBu", annot=True, fmt="g", annot_kws={"size": 20})
        plt.xlabel("predicted", fontsize=18)
        plt.ylabel("actual", fontsize=18)
        title = "Confusion Matrix for " + classifier_name
        plt.title(title, fontsize=18)
        # plt.show()
        return precision, recall, accuracy, f1
