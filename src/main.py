from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from preprocessing.corpus import build_corpus
from preprocessing.data_preparation import prepare_and_merge_datasets
from wordcloud import WordCloud

if __name__ == "__main__":
    # prepare_and_merge_datasets()
    df_dataset = pd.read_csv("data/preprocessed/dataset.csv", index_col=0)
    df_dataset = build_corpus(df_dataset[df_dataset["class"] == 0])

    counter = Counter(df_dataset["stems"].explode())
    twenty_most_common = counter.most_common(20)
    x = np.array(twenty_most_common).T[0]
    twenty_most_common_words_as_string = (" ").join(x)
    wordcloud = WordCloud().generate(twenty_most_common_words_as_string)

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis("off")
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.show()
