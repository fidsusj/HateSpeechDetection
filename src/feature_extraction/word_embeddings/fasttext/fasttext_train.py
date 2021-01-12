""" This module builds the word embeddings using fasttext """

import fasttext
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class Fasttext:
    """ Generate word embeddings with fasttext """

    def train_fasttext(self, path_to_csv_dataset_file):
        """Train fasttext model based on dataset
        Parameters:
            - path_to_csv_dataset_file: relative path to the dataset file (expects a csv file)
        """
        model = fasttext.train_unsupervised(path_to_csv_dataset_file, model="skipgram")
        model.save_model("../../../models/fasttext_model.bin")
        print(model.get_nearest_neighbors("love"))

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
        for i in range(len(x)):
            plt.scatter(x[i], y[i])
            plt.annotate(
                labels[i],
                xy=(x[i], y[i]),
                xytext=(5, 2),
                textcoords="offset points",
                ha="right",
                va="bottom",
            )
        plt.savefig("../../../analysis/fasttext_tsne_visualization")
        plt.show()


if __name__ == "__main__":
    Fasttext().train_fasttext("../../../data/preprocessed/dataset.csv")
    fasttext_model = fasttext.load_model("../../../models/fasttext_model.bin")
    Fasttext().visualize_word_embeddings_with_tsne(fasttext_model)
