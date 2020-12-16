""" This module builds the word embeddings using Word2Vec """

import multiprocessing

from gensim.models import Word2Vec


def train_word2vec(dataframe):
    """ Trains the Word2Vec model """
    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(
        min_count=20,
        window=2,
        size=300,
        sample=6e-5,
        alpha=0.03,
        min_alpha=0.0007,
        negative=20,
        workers=cores - 1,
    )
    w2v_model.build_vocab(dataframe["tokens"], progress_per=10000)
    w2v_model.train(
        dataframe["tokens"],
        total_examples=w2v_model.corpus_count,
        epochs=30,
        report_delay=1,
    )
    w2v_model.save("./model.pickle")
