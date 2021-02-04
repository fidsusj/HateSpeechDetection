""" Module containing hyperparameter search spaces for each classifier """

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

hyperparameter_search_space = {
    "random_forest": [
        {
            "classifier": [RandomForestClassifier()],
            "classifier__criterion": ["gini", "entropy"],
            "classifier__max_features": [None, "sqrt", "log2"],
            "classifier__max_depth": [None, 10],
        }
    ],
    "decision_tree": [
        {
            "classifier": [DecisionTreeClassifier()],
            "classifier__criterion": ["gini", "entropy"],
            "classifier__max_depth": [None, 2, 10],
            "classifier__min_samples_split": [2, 20, 40],
            "classifier__min_samples_leaf": [1, 5, 10],
            "classifier__max_leaf_nodes": [
                None,
                10,
                15,
            ],
            "classifier__class_weight": [None, "balanced"],
        }
    ],
    "svm": [
        {"classifier": [SVC()], "classifier__kernel": ["linear", "rbf", "sigmoid"]}
    ],
    "logistic_regression": [
        {
            "classifier": [LogisticRegression()],
            "classifier__C": [0.8, 0.9, 1.0, 1.1, 1.2],
            "classifier__solver": ["lbfgs", "sag", "saga"],
            "classifier__max_iter": [1500],
        }
    ],
}
