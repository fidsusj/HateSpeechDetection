""" Module runs all classifiers in this directory and returns a dataframe with all predictions """
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from classifiers.hyperparameters import hyperparameter_search_space
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


class ClassifierExecutor:
    """ Extract all feature and return dataframe with all features """

    def __init__(self, X_train, y_train, X_test, y_test):
        classifier_names = ["random_forest", "decision_tree"]
        classifiers = [RandomForestClassifier(), DecisionTreeClassifier()]
        classifier_tuple = zip(classifier_names, classifiers)

        self.df_evaluation_results = self.run_classifiers(
            classifier_tuple, X_train, y_train, X_test, y_test
        )

    def get_evaluation_results(self):
        """ Getter for the df_evaluation_results """
        return self.df_evaluation_results

    def run_classifiers(self, classifiers, X_train, y_train, X_test, y_test):
        """Trains different parameters, optimizes hyperparameters and tests models
        Parameters:
            classifiers: tuple containing (classifier_name, classifier_class)
            X_train: training dataset features
            y_train: training dataset labels
            X_test: test dataset features
            y_test: test dataset labels
        Return:
            df_evaluation_results: dataframe with the evaluation results of all passed classifiers
        """
        results = []
        for classifier_name, classifier in classifiers:
            classifier_pipeline = Pipeline([("classifier", classifier)])

            gridsearch = GridSearchCV(
                classifier_pipeline,
                hyperparameter_search_space[classifier_name],
                cv=5,
                verbose=0,
                n_jobs=-1,
            )
            gridsearch.fit(X_train, y_train)
            best_model = gridsearch.best_estimator_
            classifier_evaluation = self.evaluate_classifier_on_test_set(
                best_model, X_test, y_test, classifier_name
            )
            results.append(classifier_evaluation)
        df_evaluation_results = pd.DataFrame(
            data=results,
            columns=["classifier", "precision", "recall", "accuracy", "f1"],
        )
        return df_evaluation_results

    def evaluate_classifier_on_test_set(
        self, best_model, X_test, y_test, classifier_name
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
        precision, recall, accuracy, f1 = self.calculate_performance_metrices(
            y_test, y_predicted, classifier_name
        )
        return [classifier_name, precision, recall, accuracy, f1]

    def calculate_performance_metrices(self, y, y_hat, classifier_name):
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
