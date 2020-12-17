""" This module builds a dictionary based on training data containing
typical hateful and neutral words """

import pandas as pd
from feature_extraction.ngram.tfidf import TfIdf
from preprocessing.corpus import tokenization


class Dictionary:
    """ Creates a Dictionary of hateful/neutral words based on passed training data """

    def __init__(self, training_set_hate, training_set_neutral, dictionary_size):
        self.training_set_hate = training_set_hate
        self.training_set_neutral = training_set_neutral

        tfidf = TfIdf(1, "english")
        list_hate = self._transform_df_column_to_one_list(self.training_set_hate)
        list_neutral = self._transform_df_column_to_one_list(self.training_set_neutral)
        df_tfidf = tfidf.calculate_tfidf([list_hate, list_neutral])
        (
            self.dictionary_hate_speech_words,
            self.dictionary_neutral_words,
        ) = self._create_dictionary(df_tfidf, dictionary_size)

    def extract_features(self, df):
        """Extract number of special characters per data instance
        Parameters:
            df with the columns: class and content
        Return:
            passed df with new feature columns
            containing the count of the hateful/neutral words per data instance
        """
        df_tokens = tokenization(df)
        df_tokens["number_of_hateful_words"] = df_tokens["content"].apply(
            lambda cell: self._check_if_list_contains_words(
                cell, self.dictionary_hate_speech_words
            )
        )
        df_tokens["number_of_neutral_words"] = df_tokens["content"].apply(
            lambda cell: self._check_if_list_contains_words(
                cell, self.dictionary_neutral_words
            )
        )
        return df_tokens

    def _check_if_list_contains_words(self, word_list, dictionary):
        number_dictionary_matches_in_word_list = 0
        for word in dictionary:
            if word in word_list:
                number_dictionary_matches_in_word_list = (
                    number_dictionary_matches_in_word_list + word_list.count(word)
                )
        return number_dictionary_matches_in_word_list

    def _transform_df_column_to_one_list(self, df):
        content_column_as_list = df["content"].tolist()
        list_with_one_item = " ".join(content_column_as_list)
        return list_with_one_item

    def _create_dictionary(self, df_tfidf, dictionary_size):
        top_hate_speech_words_based_on_tfidf = (
            self._sort_df_by_column_and_return_top_n_items(
                df_tfidf, 0, 1, dictionary_size
            )
        )
        top_neutral_words_based_on_tfidf = (
            self._sort_df_by_column_and_return_top_n_items(
                df_tfidf, 1, 1, dictionary_size
            )
        )

        dictionary_hate_speech_words = self._get_distinct_list_elements(
            top_hate_speech_words_based_on_tfidf, top_neutral_words_based_on_tfidf
        )
        dictionary_neutral_words = self._get_distinct_list_elements(
            top_neutral_words_based_on_tfidf, top_hate_speech_words_based_on_tfidf
        )

        print(len(dictionary_hate_speech_words))
        print(len(dictionary_neutral_words))

        return dictionary_hate_speech_words, dictionary_neutral_words

    def _sort_df_by_column_and_return_top_n_items(self, df, column, axis, n):
        df_tfidf_sorted = df.sort_values(by=[column], axis=axis, ascending=False)
        return df_tfidf_sorted.columns.tolist()[:n]

    def _get_distinct_list_elements(self, list_1, list_2):
        return set(list_1) - set(list_2)


if __name__ == "__main__":
    df_dataset = pd.read_csv("../../data/preprocessed/dataset.csv", index_col=0)
    list_hate_speech = df_dataset[df_dataset["class"] == 0]
    list_neutral_speech = df_dataset[df_dataset["class"] == 1]

    my_dictionary = Dictionary(list_hate_speech, list_neutral_speech, 100)
    my_dictionary.extract_features(df_dataset)
