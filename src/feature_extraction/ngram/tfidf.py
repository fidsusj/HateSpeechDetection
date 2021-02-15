""" Module for calculating TF_IDF """

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class TfIdf:
    """Calculate TF_IDF
    Parameters:
        - passed_min_df: min_df parameter for CountCectorizer
        - passed_stop_words: list of stop words (e.g. "english")
    """

    def __init__(self, passed_min_df, passed_stop_words=None):
        self.vectorizer = CountVectorizer(
            stop_words=passed_stop_words, min_df=passed_min_df
        )

    def calculate_tfidf(self, list_of_docs):
        """Calculate tfidf for a given list of documents
        Parameters:
            - list_of_docs: list containing strings foreach document
              (e.g. ["document 1 hello", "document 2"]
        Return:
            - df_tf_idf: dataframe with tfidf matrix
        """
        tf = self._calculate_tf(list_of_docs)
        df_tf_idf = self._calc_tfidf(tf)
        return df_tf_idf

    def _calculate_tf(self, list_of_docs):
        tf = self.vectorizer.fit_transform(list_of_docs)
        return tf

    def _calc_tfidf(self, tf):
        transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        X = transformer.fit_transform(tf)
        df_tf_idf = pd.DataFrame(
            X.toarray(), columns=self.vectorizer.get_feature_names()
        )
        return df_tf_idf

    @staticmethod
    def visualize_as_heatmap(df_tfidf, top_n_items):
        """Visualize tfidf
        Parameters:
            - df_tfidf: tfidf matrix
            - top_n_items: number how many items should be visualized
        Return:
            - saves image
        """
        df_tfidf_sorted = df_tfidf.sort_values(by=[0, 1], axis=1, ascending=False)
        df_tfidf_sorted_top_n = df_tfidf_sorted[df_tfidf_sorted.columns[:top_n_items]]
        df_tfidf_sorted_top_n.index = ["hate", "neutral"]

        ax = plt.subplots(figsize=(17, 6))
        sns.heatmap(
            df_tfidf_sorted_top_n,
            annot=True,
            cbar=True,
            ax=ax,
            xticklabels=df_tfidf_sorted.columns[:top_n_items],
            cmap="YlGnBu",
        )
        plt.savefig("../../../docs/milestone/figures/Heatmap_tfidf.png")
        plt.show()
