from unittest import TestCase

from src.classifiers.classifier_executor import ClassifierExecutor


class TestClassifierExecutor(TestCase):
    datasets = {
        "raw_datasets": {
            "unchanged": {"X_train": 0, "y_train": 1, "X_test": 2, "y_test": 3},
            "undersampled": {"X_train": 4, "y_train": 5, "X_test": 6, "y_test": 7},
        },
        "extracted_datasets": {
            "unchanged": {"X_train": 8, "y_train": 9, "X_test": 10, "y_test": 11},
            "undersampled": {"X_train": 12, "y_train": 13, "X_test": 14, "y_test": 15},
            "oversampled": {"X_train": 16, "y_train": 17, "X_test": 18, "y_test": 19},
        },
    }
    executor = object.__new__(ClassifierExecutor)

    def test__create_run_parameters(self):
        datasets = {"Hello": "world"}
        expected = [
            ("Classifier1", "ONE", {"Hello": "world"}),
            ("Classifier2", "ONE", {"Hello": "world"}),
            ("Classifier1", "TWO", {"Hello": "world"}),
            ("Classifier2", "TWO", {"Hello": "world"}),
        ]
        real = self.executor._create_run_parameters(
            ["Classifier1", "Classifier2"], ["ONE", "TWO"], datasets
        )
        print(real)
        self.assertEqual(real, expected)

    def test__extract_train_and_test(self):
        expected = 0, 1, 2, 3
        real = self.executor._extract_train_and_test(
            self.datasets, "raw_datasets", "unchanged"
        )
        self.assertEquals(real, expected)
        expected = 4, 5, 6, 7
        real = self.executor._extract_train_and_test(
            self.datasets, "raw_datasets", "undersampled"
        )
        self.assertEquals(real, expected)
        expected = 8, 9, 10, 11
        real = self.executor._extract_train_and_test(
            self.datasets, "extracted_datasets", "unchanged"
        )
        self.assertEquals(real, expected)
        expected = 12, 13, 14, 15
        real = self.executor._extract_train_and_test(
            self.datasets, "extracted_datasets", "undersampled"
        )
        self.assertEquals(real, expected)
        expected = 16, 17, 18, 19
        real = self.executor._extract_train_and_test(
            self.datasets, "extracted_datasets", "oversampled"
        )
        self.assertEquals(real, expected)

    def test__calculate_performance_metrices(self):
        expected = (1.0, 0.5, 0.6666666666666666, 0.6666666666666666)
        y = [1, 0, 1]
        y_hat = [1, 0, 0]
        classifier_name = "MyTestClassifier"
        real = self.executor._calculate_performance_metrices(y, y_hat, classifier_name)
        print(real)
        self.assertEquals(real, expected)
