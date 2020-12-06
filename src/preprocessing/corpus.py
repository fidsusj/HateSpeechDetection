import spacy
from nltk.stem import PorterStemmer


def build_corpus(dataframe):
    nlp = spacy.load("en")
    nlp.disable_pipes("tagger", "parser", "ner")
    stopwords = spacy.lang.en.stop_words.STOP_WORDS
    stemmer = PorterStemmer()

    dataframe["content"] = (
        dataframe["content"].str.lower().str.replace("[^a-zA-Z ]", "")
    )

    def tokenize(input_string):
        doc = nlp(input_string)
        tokens = []
        for token in doc:
            tokens.append(token.text)
        return tokens

    dataframe["all_tokens"] = dataframe["content"].apply(lambda cell: tokenize(cell))

    def remove_stopwords(input_list_of_tokens):
        return [token for token in input_list_of_tokens if not token in stopwords]

    dataframe["tokens_without_stopwords"] = dataframe["all_tokens"].apply(
        lambda cell: remove_stopwords(cell)
    )

    def remove_single_characters(input_list_of_tokens):
        return [token for token in input_list_of_tokens if len(token) >= 3]

    dataframe["cleaned_tokens"] = dataframe["tokens_without_stopwords"].apply(
        lambda cell: remove_single_characters(cell)
    )

    def perform_stemming(input_list_of_tokens):
        stems = []
        for token in input_list_of_tokens:
            stems.append(stemmer.stem(token))
        return stems

    dataframe["stems"] = dataframe["cleaned_tokens"].apply(
        lambda cell: perform_stemming(cell)
    )

    return dataframe
