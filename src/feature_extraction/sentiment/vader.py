""" This module builds the sentiment polarity feature """

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")


class Vader:
    """ Extract polarity based on nltk vader """

    def extract_features(self, df):
        """Extract polarity score
        Parameters:
            df with the columns: class and content
        Return:
            passed df with new feature columns containing the polarity
        """
        vader = SentimentIntensityAnalyzer()
        df["sentiment"] = df["content"].apply(
            lambda x: vader.polarity_scores(x)["compound"]
        )
        return df
