""" This module builds the word embeddings using fasttext """

import fasttext
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from path_helper import get_project_root
from sklearn.manifold import TSNE


class Fasttext:
    """ Generate word embeddings with fasttext """

    def __init__(self):
        self.train_fasttext(str(get_project_root()) + "/data/preprocessed/dataset.csv")

    def train_fasttext(self, path_to_csv_dataset_file):
        """Train fasttext model based on dataset
        Parameters:
            - path_to_csv_dataset_file: relative path to the dataset file (expects a csv file)
        """
        model = fasttext.train_unsupervised(path_to_csv_dataset_file, model="skipgram")
        model.save_model(str(get_project_root()) + "/models/fasttext_model.bin")

    def extract_features(self, df, visualize=False):
        """Extract vector representation of the data instance based on word embeddings trained by fasttext
        Parameters:
            df with the column containing the tokens of each data instance
        Return:
            passed df with new feature column containing a vector (mean of the word embeddings of all tokens)
        """
        model = fasttext.load_model(
            str(get_project_root()) + "/models/fasttext_model.bin"
        )
        df_fasttext_vector = pd.DataFrame()
        df_fasttext_vector["fasttext_word_embeddings_vector"] = df["tokens"].apply(
            lambda cell: self.get_vector_of_data_instance(model, cell)
        )
        df_fasttext_vector = pd.DataFrame(
            df_fasttext_vector["fasttext_word_embeddings_vector"].values.tolist()
        )
        titles = ["fasttext_word_embeddings_vector_" + str(i) for i in range(100)]
        df_fasttext_vector.columns = titles
        df = pd.concat([df_fasttext_vector, df], axis=1)
        if visualize:
            self.visualize_word_embeddings_with_tsne(model)
        return df

    def get_vector_of_data_instance(self, model, list_of_tokens):
        """For each word in data instance get word vector and return mean over all word vectors of the data instance
        Parameters:
            model: the fasttext model
            list_of_tokens: [token1, token2] tokens of one data instance
        Return:
            one vector (mean of the word embeddings of all tokens of the data instance)
        """
        word_vectors = []
        for token in list_of_tokens:
            word_vectors.append(model.get_word_vector(token))
        if len(word_vectors) == 0:
            mean_over_all_word_vectors = np.zeros(100)
        else:
            mean_over_all_word_vectors = np.mean(np.array(word_vectors), axis=0)
        return mean_over_all_word_vectors

    def visualize_word_embeddings_with_tsne(self, model):
        """Creates an TSNE model and plots it
        Parameters:
            - model: fasttext model
        """
        labels = []
        tokens = []

        for word in model.words:
            tokens.append(model[word])
            labels.append(word)

        tsne_model = TSNE(
            perplexity=40, n_components=2, init="pca", n_iter=2500, random_state=23
        )
        new_values = tsne_model.fit_transform(tokens)

        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])

        plt.figure(figsize=(16, 16))
        for index, x in enumerate(x):
            plt.scatter(x[index], y[index])
            plt.annotate(
                labels[index],
                xy=(x[index], y[index]),
                xytext=(5, 2),
                textcoords="offset points",
                ha="right",
                va="bottom",
            )
        plt.savefig(str(get_project_root()) + "/analysis/fasttext_tsne_visualization")
        plt.show()


if __name__ == "__main__":
    Fasttext().train_fasttext("../../../data/preprocessed/dataset.csv")
    fasttext_model = fasttext.load_model("../../../models/fasttext_model.bin")
    Fasttext().visualize_word_embeddings_with_tsne(fasttext_model)
