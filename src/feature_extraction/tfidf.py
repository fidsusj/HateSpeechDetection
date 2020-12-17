import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class TfIdf:
    def __init__(self):
        # TODO: adapt and pass parameters
        self.vectorizer = CountVectorizer(stop_words="english", min_df=2)

    def calculate_tfidf(self, list_of_docs):
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


if __name__ == "__main__":
    df_dataset = pd.read_csv("../data/preprocessed/dataset.csv", index_col=0)
    list_hate_speech = df_dataset[df_dataset["class"] == 0]["content"].tolist()
    list_neutral_speech = df_dataset[df_dataset["class"] == 1]["content"].tolist()

    list_hate_speech_as_one_doc = " ".join(list_hate_speech)
    list_neutral_speech_as_one_doc = " ".join(list_neutral_speech)

    list_of_docs = [list_hate_speech_as_one_doc, list_neutral_speech_as_one_doc]

    df_tfifd = TfIdf().calculate_tfidf(list_of_docs)

    df_tfidf_sorted = df_tfifd.sort_values(by=[0, 1], axis=1, ascending=False)

    n = 20
    df_tfidf_sorted_top_n = df_tfidf_sorted[df_tfidf_sorted.columns[:n]]

    df_tfidf_sorted_top_n.index = ["hate", "neutral"]

    fig, ax = plt.subplots(figsize=(17, 6))
    sns.heatmap(
        df_tfidf_sorted_top_n,
        annot=True,
        cbar=True,
        ax=ax,
        xticklabels=df_tfidf_sorted.columns[:n],
        cmap="YlGnBu",
    )
    plt.show()
    print("Ende")
