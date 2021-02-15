from unittest import TestCase

import pandas as pd
from classifiers.input_data import InputData


class TestInputData(TestCase):
    def test_constructor(self):
        raw_text_features_list = ["Hello", "Hello2", "Hello3", "Hello4"]
        raw_text_features_list_df = pd.DataFrame(
            data=raw_text_features_list, columns=["text"]
        )
        raw_text_labels = [[0], [1], [0], [1]]
        extracted_features = [[1, 0], [3, 4], [5, 6], [7, 8]]
        labels = [[0], [1], [0], [1]]
        input_data = InputData(
            raw_text_features_list_df["text"],
            raw_text_labels,
            extracted_features,
            labels,
        )

        self.assertTrue(
            any(
                i in raw_text_features_list
                for i in input_data.datasets["raw_datasets"]["unchanged"][
                    "X_train"
                ].values.tolist()
            )
        )
        self.assertTrue(
            any(
                i in raw_text_labels
                for i in input_data.datasets["raw_datasets"]["unchanged"]["y_train"]
            )
        )
        self.assertTrue(
            any(
                i in raw_text_features_list
                for i in input_data.datasets["raw_datasets"]["unchanged"][
                    "X_test"
                ].values.tolist()
            )
        )
        self.assertTrue(
            any(
                i in raw_text_labels
                for i in input_data.datasets["raw_datasets"]["unchanged"]["y_test"]
            )
        )

        self.assertTrue(
            any(
                i in extracted_features
                for i in input_data.datasets["extracted_datasets"]["unchanged"][
                    "X_train"
                ]
            )
        )
        self.assertTrue(
            any(
                i in labels
                for i in input_data.datasets["extracted_datasets"]["unchanged"][
                    "y_train"
                ]
            )
        )
        self.assertTrue(
            any(
                i in extracted_features
                for i in input_data.datasets["extracted_datasets"]["unchanged"][
                    "X_test"
                ]
            )
        )
        self.assertTrue(
            any(
                i in labels
                for i in input_data.datasets["extracted_datasets"]["unchanged"][
                    "y_test"
                ]
            )
        )

    def test_get_datasets(self):
        input_data = object.__new__(InputData)
        datasets = [3, 4, 5]
        input_data.datasets = datasets
        self.assertEqual(datasets, input_data.get_datasets())

    def test_split(self):
        features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        labels = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
        input_data = object.__new__(InputData)
        X_train, X_test, y_train, y_test = input_data.split(features, labels)
        self.assertTrue(len(X_train) == 8)
        self.assertTrue(len(y_train) == 8)
        self.assertTrue(len(X_test) == 2)
        self.assertTrue(len(y_test) == 2)
