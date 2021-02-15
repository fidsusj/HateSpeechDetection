""" Module builds a dictionary based on training data containing
typical hateful and neutral words """

from feature_extraction.ngram.tfidf import TfIdf


class Dictionary:
    """Creates a Dictionary of hateful/neutral words based on passed training data
    and returns the dataframe with the new feature columns for the number of hate/neutral tokens
    """

    def __init__(self):
        """Set dictionary parameters:
        - dictionary_size: number
        """
        self.dictionary_size = 100
        self.dictionary_hate_speech_words = []
        self.dictionary_neutral_words = []

    def extract_features(self, df):
        """Extract number of hateful/neutral tokens per data instance
        Parameters:
            df with the columns: class and content
        Return:
            passed df with new feature columns
            containing the count of the hateful/neutral words per data instance
        """
        self.transform_data_and_create_dictionary(df)
        df["number_of_hateful_words"] = df["content"].apply(
            lambda cell: self._check_if_list_contains_words(
                cell, self.dictionary_hate_speech_words
            )
        )
        df["number_of_neutral_words"] = df["content"].apply(
            lambda cell: self._check_if_list_contains_words(
                cell, self.dictionary_neutral_words
            )
        )
        return df

    def transform_data_and_create_dictionary(self, df):
        """Transforms the input dataframe to the correct form
        and builds the dictionary
        Parameters:
            df with the columns: class and content
        Return:
            no return values
            but the two dictionaries (hate, neutral) are stored as instance variables
        """
        training_set_hate = df[df["class"] == 0]
        training_set_neutral = df[df["class"] == 1]

        tfidf = TfIdf(1, "english")
        list_hate = self._transform_df_column_to_one_list(training_set_hate)
        list_neutral = self._transform_df_column_to_one_list(training_set_neutral)
        df_tfidf = tfidf.calculate_tfidf([list_hate, list_neutral])
        (
            self.dictionary_hate_speech_words,
            self.dictionary_neutral_words,
        ) = self._create_dictionary(df_tfidf, self.dictionary_size)

    @staticmethod
    def _check_if_list_contains_words(word_list, dictionary):
        number_dictionary_matches_in_word_list = 0
        for word in dictionary:
            if word in word_list:
                number_dictionary_matches_in_word_list = (
                    number_dictionary_matches_in_word_list + word_list.count(word)
                )
        return number_dictionary_matches_in_word_list

    @staticmethod
    def _transform_df_column_to_one_list(df):
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

        return dictionary_hate_speech_words, dictionary_neutral_words

    @staticmethod
    def _sort_df_by_column_and_return_top_n_items(df, column, axis, n):
        df_tfidf_sorted = df.sort_values(by=[column], axis=axis, ascending=False)
        return df_tfidf_sorted.columns.tolist()[:n]

    @staticmethod
    def _get_distinct_list_elements(list_1, list_2):
        return set(list_1) - set(list_2)
