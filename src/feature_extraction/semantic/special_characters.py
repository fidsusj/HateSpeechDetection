""" Module for feature extraction based on special characters (e.g. ? ! .) """

import matplotlib.pyplot as plt
import numpy as np
from path_helper import get_project_root


class SpecialCharacters:
    """ Extract list of special characters (list is set in constructor) """

    def __init__(self):
        self.list_of_special_characters = {
            "exclamation_mark": "!",
            "question_mark": "?",
            "full_stop_mark": ".",
        }

    def extract_features(self, df, visualize=True):
        """Extract number of special characters per data instance
        Parameters:
            df with the columns: class and content
        Return:
            passed df with new feature columns containing the count of the special character
        """
        for key in self.list_of_special_characters:
            df = self._count_number_of_special_characters(
                key, self.list_of_special_characters[key], df
            )
        if visualize:
            self.visualize_special_characters(df)
        return df

    def _count_number_of_special_characters(self, character_name, character, df):
        feature_name = "number_of_" + character_name
        df[feature_name] = df["content"].apply(
            lambda cell: self._count_character(cell, character)
        )
        return df

    @staticmethod
    def _count_character(sentence, character):
        return sentence.count(character)

    def visualize_special_characters(self, df):
        """Visualizes the number of special characters as bar plot
        Parameters:
            df: dataframe with the extracted features for special characters
        Return:
            stores barplots in analysis folder
        """
        df_hate_speech = df[df["class"] == 0]
        df_neutral_speech = df[df["class"] == 1]
        for character in self.list_of_special_characters:
            hate_bincount = self._calculate_bincount_of_special_character(
                df_hate_speech, character
            )
            neutral_bincount = self._calculate_bincount_of_special_character(
                df_neutral_speech, character
            )

            hate_bincount_summarized = self._summarize_bincount_data(hate_bincount)
            neutral_bincount_summarized = self._summarize_bincount_data(
                neutral_bincount
            )

            x = np.arange(11)
            plt.bar(x + 0.0, hate_bincount_summarized, color="r", width=0.2)
            plt.bar(x + 0.2, neutral_bincount_summarized, color="b", width=0.2)
            x_ticks = [str(x) for x in range(10)]
            x_ticks.append(">10")
            plt.xticks(x, x_ticks)
            plt.title(
                "Number of data instances with number of "
                + self.list_of_special_characters[character]
            )
            plt.xlabel("Number of " + character)
            plt.ylabel("Number of data instances")
            plt.legend(["hate speech", "neutral speech"])
            plt.savefig(
                str(get_project_root())
                + "/analysis/features/semantic/barchart_special_character_"
                + character
            )

    def _calculate_bincount_of_special_character(self, df, character):
        return np.bincount(np.array(df["number_of_" + character]))

    def _summarize_bincount_data(self, array):
        return np.append(array[:10], array[10:].sum())
