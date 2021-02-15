from collections import Counter
from unittest import TestCase

from preprocessing.class_balancer import ClassBalancer


class TestClassBalancer(TestCase):
    features = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8]]
    labels = [1, 1, 0, 0, 0, 0, 0, 0]

    balancer = ClassBalancer(features, labels)

    def test_undersampling(self):
        undersampled_features, undersampled_labels = self.balancer.undersample()
        expected_counter = Counter({0: 2, 1: 2})
        counter = Counter(undersampled_labels)
        self.assertEqual(expected_counter, counter)

    def test_oversampling(self):
        oversampled_features, oversampled_labels = self.balancer.oversample(
            k_neighbors=1
        )
        expected_counter = Counter({0: 6, 1: 6})
        counter = Counter(oversampled_labels)
        self.assertEqual(expected_counter, counter)
