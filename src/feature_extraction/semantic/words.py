"""
Module for feature extraction based on words
(e.g. all-capitalized words, interjections, number of words)
"""

from collections import Counter

import nltk

nltk.download("averaged_perceptron_tagger")
nltk.download("punkt")


class Words:
    """
    Extract features based on words
    (all-capitalized, interjections, number of words)
    """

    def extract_features(self, df):
        """Extract features
        Parameters:
            df with the columns: class and content
        Return:
            passed df with new feature columns containing the count of the special character
        """

        # get Parts of Speech (POS) Tagging from nltk and counting interjections ("UH")
        tag_list = nltk.pos_tag_sents(df["content"].apply(nltk.word_tokenize).tolist())
        count_list = [Counter(tag for word, tag in tags) for tags in tag_list]
        df["number_of_interjections"] = [counts.get("UH", 0) for counts in count_list]

        # count number of all-capitalized words
        df["number_of_all_caps_words"] = [
            len([word for word in text.split() if word.isupper()])
            for text in df["original_content"]
        ]

        # count number of quotation mark characters (approximation for number of quotes)
        quote_pattern = r"(\")|('')|(``)"
        df["number_of_quotations"] = df["content"].str.count(quote_pattern)

        return df
