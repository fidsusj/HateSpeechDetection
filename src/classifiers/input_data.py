""" Class holding all datasets [(raw or extracted),(unchanged, oversampled, undersampled)]
for inserting them into the classifiers """

from preprocessing.class_balancer import ClassBalancer
from sklearn.model_selection import train_test_split


class InputData:
    """ Class holds input data for the classifiers """

    def __init__(self, raw_text_features, raw_text_labels, extracted_features, labels):

        raw_data_balancer = ClassBalancer(
            [[i] for i in raw_text_features.values.tolist()], raw_text_labels
        )
        extracted_data_balancer = ClassBalancer(extracted_features, labels)
        # raw
        X_train, X_test, y_train, y_test = self.split(
            raw_text_features, raw_text_labels
        )
        undersampled_x, undersampled_y = raw_data_balancer.undersample()
        u_X_train, u_X_test, u_y_train, u_y_test = self.split(
            undersampled_x, undersampled_y
        )
        # extracted
        e_X_train, e_X_test, e_y_train, e_y_test = self.split(
            extracted_features, labels
        )
        undersampled_x, undersampled_y = extracted_data_balancer.undersample()
        eu_X_train, eu_X_test, eu_y_train, eu_y_test = self.split(
            undersampled_x, undersampled_y
        )
        oversampled_x, oversampled_y = extracted_data_balancer.oversample()
        eo_X_train, eo_X_test, eo_y_train, eo_y_test = self.split(
            oversampled_x, oversampled_y
        )

        self.datasets = {
            "raw_datasets": {
                "unchanged": {
                    "X_train": X_train,
                    "y_train": y_train,
                    "X_test": X_test,
                    "y_test": y_test,
                },
                "undersampled": {
                    "X_train": u_X_train,
                    "y_train": u_y_train,
                    "X_test": u_X_test,
                    "y_test": u_y_test,
                },
            },
            "extracted_datasets": {
                "unchanged": {
                    "X_train": e_X_train,
                    "y_train": e_y_train,
                    "X_test": e_X_test,
                    "y_test": e_y_test,
                },
                "undersampled": {
                    "X_train": eu_X_train,
                    "y_train": eu_y_train,
                    "X_test": eu_X_test,
                    "y_test": eu_y_test,
                },
                "oversampled": {
                    "X_train": eo_X_train,
                    "y_train": eo_y_train,
                    "X_test": eo_X_test,
                    "y_test": eo_y_test,
                },
            },
        }

    def get_datasets(self):
        """ Getter for datasets """
        return self.datasets

    def split(self, features, labels):
        """ Splits features and labels into train and test """
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=0.2,
            random_state=42,
        )
        return X_train, X_test, y_train, y_test
