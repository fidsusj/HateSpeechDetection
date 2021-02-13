""" This module builds the corpus """

import nltk
from nltk.stem.snowball import SnowballStemmer
from spacy.lang.en import English

nltk.download("averaged_perceptron_tagger")


def preprocessing(dataframe):
    """ Makes all words lower case, removes the emojis and all other irrelevant characters """
    dataframe.loc[:, "content"] = dataframe.loc[:, "content"].str.replace("&.*?;", "")
    dataframe.loc[:, "content"] = dataframe.loc[:, "content"].str.replace(
        r"[^a-zA-Z\s.:!?/#\"]", ""
    )
    dataframe["original_content"] = dataframe["content"]
    dataframe.loc[:, "content"] = dataframe.loc[:, "content"].str.lower()
    return dataframe


def tokenization(dataframe):
    """ Tokenization, POS tagging, stemming, stop word removal, white space removal and punctuation removal """
    nlp = English()
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    stemmer = SnowballStemmer("english")

    dataframe.loc[:, "tokens"] = dataframe.loc[:, "content"].apply(
        lambda cell: [
            token.lemma_
            for token in tokenizer(cell)
            if token.is_stop is False
            and token.is_space is False
            and token.is_punct is False
        ]
    )
    dataframe.loc[:, "tokens"] = dataframe.loc[:, "tokens"].apply(
        lambda cell: [token for token in cell if len(token) >= 2]
    )
    dataframe["pos"] = dataframe["content"].apply(
        lambda cell: [
            tag for (word, tag) in nltk.pos_tag(cell.split())
        ]  # spacy pos tagger does work properly, use content as important words like "I" or "He" are removed in
        # tokens (stopwords)
    )
    dataframe.loc[:, "stems"] = dataframe.loc[:, "tokens"].apply(
        lambda cell: [stemmer.stem(token) for token in cell]
    )
    return dataframe


def build_corpus(dataframe):
    """ Builds the corpus """
    return tokenization(preprocessing(dataframe))
