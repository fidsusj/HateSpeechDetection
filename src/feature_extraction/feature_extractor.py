""" Module extracts all features in this package and returns one dataframe with all feature """

from feature_extraction.ngram.dictionary import Dictionary
from feature_extraction.semantic.special_characters import SpecialCharacters
from feature_extraction.semantic.words import Words


class FeatureExtractor:
    """ Extract all feature and return dataframe with all features """

    def __init__(self, df):
        feature_class_names = [SpecialCharacters, Dictionary, Words]
        self.df_with_all_extracted_features = self._extract_all_features(
            df, feature_class_names
        )

    def get_df_with_all_features(self):
        """Getter for the instance variable df_with_all_extracted_features"""
        return self.df_with_all_extracted_features

    def _extract_all_features(self, df, feature_class_names):
        """Extract all number of special characters per data instance
        Parameters:
            df: dataframe from which the features should be extracted
            feature_class_names: list of class names of the feature classes
        Return:
            df with all extracted features
        """
        for feature_class_name in feature_class_names:
            feature = feature_class_name()
            feature.extract_features(df)
        return df
