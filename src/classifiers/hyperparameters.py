""" Module containing hyperparameter seacrh spaces for each classifier """

from sklearn.ensemble import RandomForestClassifier
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
}
