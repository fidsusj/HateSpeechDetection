""" This module builds the sentiment polarity feature """

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")


def get_sentiment_features(df):
    """ Calculates polarity scores """
    vader = SentimentIntensityAnalyzer()
    df["sentiment"] = df["content"].apply(
        lambda x: vader.polarity_scores(x)["compound"]
    )
    return df
