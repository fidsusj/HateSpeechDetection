""" This module builds the corpus """

from spacy.lang.en import English


def preprocessing(dataframe):
    """ Makes all words lower case, removes the emojis and all other irrelevant characters """
    dataframe.loc[:, "content"] = dataframe.loc[:, "content"].str.lower()
    dataframe.loc[:, "content"] = dataframe.loc[:, "content"].str.replace("&.*?;", "")
    dataframe.loc[:, "content"] = dataframe.loc[:, "content"].str.replace(
        r"[^a-zA-Z\s.:!?/#]", ""
    )
    return dataframe


def tokenization(dataframe):
    """ Tokenization, stop word removal, white space removal and punctuation removal """
    nlp = English()
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
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
    return dataframe


def build_corpus(dataframe):
    """ Builds the corpus """
    return tokenization(preprocessing(dataframe))
