""" This module builds the sentiment polarity feature """

from nltk.sentiment.vader import SentimentIntensityAnalyzer


def get_sentiment_features(df):
    """ Calculates polarity scores """
    vader = SentimentIntensityAnalyzer()
    df["sentiment"] = df["content"].apply(
        lambda x: vader.polarity_scores(x)["compound"]
    )
    return df
