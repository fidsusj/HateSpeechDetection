""" This is the main module """

import pandas as pd
from gensim.models import Word2Vec
from preprocessing.corpus import build_corpus

# Configs
pd.options.mode.chained_assignment = None

if __name__ == "__main__":
    df_dataset = pd.read_csv("data/preprocessed/dataset.csv", index_col=0)
    df_dataset = build_corpus(df_dataset)
    w2v_model = Word2Vec.load("./model.pickle")
    print(w2v_model.wv.most_similar("niggas"))
