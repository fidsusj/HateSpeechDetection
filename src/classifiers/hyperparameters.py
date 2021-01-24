""" Module containing hyperparameter seacrh spaces for each classifier """

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

hyperparameter_search_space = {
    "random_forest": [
        {
            "classifier": [RandomForestClassifier()],
            "classifier__n_estimators": [10, 20, 100],
            "classifier__min_samples_split": [20],
        }
    ],
    "decision_tree": [
        {
            "classifier": [DecisionTreeClassifier()],
            "classifier__criterion": ["gini", "entropy"],
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
