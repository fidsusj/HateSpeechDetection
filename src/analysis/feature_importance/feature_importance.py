""" Module extracts the feature importance scores of each feature for every classifier """

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class FeatureImportance:
    """ Class to extract the feature importance scores. Uses optimal hyperparameters from GridSearch results """

    def __init__(self, features, labels, feature_names):
        X_train, _, y_train, _ = train_test_split(
            features,
            labels,
            test_size=0.2,
            random_state=42,
        )

        # self.classifier_SVC = SVC(kernel="linear")
        self.classifier_logistic_regression = LogisticRegression(
            max_iter=1500, C=1.0, solver="lbfgs"
        )
        self.classifier_random_forest = RandomForestClassifier(
            min_samples_split=20, n_estimators=100
        )
        self.classifier_decision_tree = DecisionTreeClassifier(criterion="entropy")
        self.feature_names = feature_names

        # self.classifier_SVC.fit(X_train, y_train)
        self.classifier_logistic_regression.fit(X_train, y_train)
        self.classifier_random_forest.fit(X_train, y_train)
        self.classifier_decision_tree.fit(X_train, y_train)

    def get_importance_scores(self):
        """ Prints feature importance scores """

        data = np.array(
            [
                # self.classifier_SVC.coef_[0],
                self.classifier_logistic_regression.coef_[0],
                self.classifier_random_forest.feature_importances_,
                self.classifier_decision_tree.feature_importances_,
            ]
        )

        df = pd.DataFrame(
            data=data,
            index=["Logistic Regression", "Random Forest", "Decision Tree"],
            columns=self.feature_names,
        )
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(df)
