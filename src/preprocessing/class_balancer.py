""" Module handles class balancing, i.e. oversampling using SMOTE and undersampling """

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


class ClassBalancer:
    """ balances classes by oversampling or undersampling and returning the balanced data sets """

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def undersample(self, random_state=None):
        """executes undersampling via RandomUnderSampler

        returns the undersampled features and labels
        """
        undersampler = RandomUnderSampler(random_state=random_state)
        undersampled_features, undersampled_labels = undersampler.fit_resample(
            self.features, self.labels
        )
        return undersampled_features, undersampled_labels

    def oversample(self, k_neighbors=6):
        """execute oversampling using SMOTE ("Synthetic Minority Over-sampling Technique")

        returns the oversampled features and labels
        """
        oversampler = SMOTE(k_neighbors=k_neighbors)
        oversampled_features, oversampled_labels = oversampler.fit_resample(
            self.features, self.labels
        )
        return oversampled_features, oversampled_labels
