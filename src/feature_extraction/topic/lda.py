""" Module extracts numerical topics of each document with LDA """


import gensim.corpora as corpora
import numpy as np
import pandas as pd
from gensim.models import LdaMulticore

from src.preprocessing.corpus import build_corpus


class LDATopic:
    """ Class to extract the LDA topics """

    def __init__(self, num_topics=2, num_workers=4):
        self.num_topics = num_topics
        self.num_workers = num_workers

    def extract_features(self, df):
        """ Extracts LDA topics """
        id2word = corpora.Dictionary(df["tokens"])
        corpus = [id2word.doc2bow(doc) for doc in df["tokens"]]
        lda_model = LdaMulticore(
            corpus=corpus,
            id2word=id2word,
            num_topics=self.num_topics,
            workers=self.num_workers,
        )
        df["topic"] = df["tokens"].apply(
            lambda tokens: np.argmax(
                [prob for (topic, prob) in lda_model[id2word.doc2bow(tokens)]]
            )
        )


if __name__ == "__main__":
    df_dataset = pd.read_csv("../../data/preprocessed/dataset.csv", index_col=0)
    df_dataset = build_corpus(df_dataset)
    lda_topic_extractor = LDATopic(
        2,
    )
    lda_topic_extractor.extract_features(df_dataset)
    print(df_dataset)
