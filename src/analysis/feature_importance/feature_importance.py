""" Module extracts the feature importance scores of each feature for every classifier """

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# from sklearn.svm import SVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class FeatureImportance:
    """ Class to extract the feature importance scores. Uses optimal hyperparameters from GridSearch results """

    def __init__(self, features, labels, feature_names):
        self.skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        self.features = features
        self.labels = labels
        self.feature_names = feature_names
        self.feature_importances = np.zeros((10, 4, 16))

    def get_importance_scores(self):
        """ Prints feature importance scores """

        index = 0
        for train_index, _ in self.skf.split(self.features, self.labels):
            classifier_SVC = SVC(kernel="linear")
            classifier_logistic_regression = LogisticRegression(
                max_iter=1500, C=1.2, solver="lbfgs"
            )
            classifier_random_forest = RandomForestClassifier(
                criterion="entropy", max_depth=10, max_features="log2"
            )
            classifier_decision_tree = DecisionTreeClassifier(
                max_leaf_nodes=15, min_samples_leaf=10
            )

            classifier_SVC.fit(
                self.features.iloc[train_index], self.labels.iloc[train_index]
            )
            classifier_logistic_regression.fit(
                self.features.iloc[train_index], self.labels.iloc[train_index]
            )
            classifier_random_forest.fit(
                self.features.iloc[train_index], self.labels.iloc[train_index]
            )
            classifier_decision_tree.fit(
                self.features.iloc[train_index], self.labels.iloc[train_index]
            )

            self.feature_importances[index] = [
                classifier_SVC.coef_[0],
                classifier_logistic_regression.coef_[0],
                classifier_random_forest.feature_importances_,
                classifier_decision_tree.feature_importances_,
            ]
            index += 1

        df_mean = pd.DataFrame(
            data=self.feature_importances.mean(axis=0),
            index=["SVM", "Logistic Regression", "Random Forest", "Decision Tree"],
            columns=self.feature_names,
        )
        df_max = pd.DataFrame(
            data=self.feature_importances.max(axis=0),
            index=["SVM", "Logistic Regression", "Random Forest", "Decision Tree"],
            columns=self.feature_names,
        )
        df_min = pd.DataFrame(
            data=self.feature_importances.min(axis=0),
            index=["SVM", "Logistic Regression", "Random Forest", "Decision Tree"],
            columns=self.feature_names,
        )

        df_mean.to_csv(
            "src/analysis/feature_importance/out_mean.zip",
            index=False,
            compression=dict(method="zip", archive_name="out_mean.csv"),
        )
        df_max.to_csv(
            "src/analysis/feature_importance/out_max.zip",
            index=False,
            compression=dict(method="zip", archive_name="out_max.csv"),
        )
        df_min.to_csv(
            "src/analysis/feature_importance/out_min.zip",
            index=False,
            compression=dict(method="zip", archive_name="out_min.csv"),
        )
