import re

import pandas as pd


def prepare_and_merge_datasets():
    dataset_csv = open("data/preprocessed/dataset.csv", encoding="utf-8", mode="w")
    df_first_dataset = _prepare_hate_speech_and_offensive_language()
    df_second_dataset = _prepare_hate_speech_dataset()

    df_dataset = pd.concat([df_first_dataset, df_second_dataset], ignore_index=True)

    dataset_csv.write(df_dataset.to_csv())
    dataset_csv.close()


def _prepare_hate_speech_and_offensive_language():
    df_dataset = _create_df_and_drop_columns(
        "data/original/hate-speech-and-offensive-language/labeled_data.csv",
        0,
        ["count", "hate_speech", "offensive_language", "neither"],
    )
    df_dataset = _filter_and_format_hate_speech_and_offensive_language(df_dataset)
    df_dataset = _data_preparation(df_dataset)
    return df_dataset


def _create_df_and_drop_columns(path_to_csv, pd_index_col, list_columns_to_be_dropped):
    df = pd.read_csv(path_to_csv, index_col=pd_index_col)
    df.drop(list_columns_to_be_dropped, axis=1, inplace=True)
    return df


def _filter_and_format_hate_speech_and_offensive_language(df):
    df = _drop_all_offensive_language_documents(df)
    df = set_neither_to_class_label_1(df)
    df.rename(columns={"tweet": "content"}, inplace=True)
    return df


def _drop_all_offensive_language_documents(df):
    df = df.drop(df[df["class"] == 1].index)
    return df


def set_neither_to_class_label_1(df):
    df.loc[df["class"] == 2, "class"] = 1
    return df


def _data_preparation(df):
    for i, row in df.iterrows():
        df.at[i, "content"] = re.sub("&.*?;", "", row["content"])  # delete all emojis
        if re.search(r"\"[^\"].*\"", row["content"]):
            df = df.drop(index=i)  # delete all tweet referencing tweets
        elif re.search(r"\"", row["content"]):
            df = df.drop(index=i)  # delete all citing tweets
    return df


def _prepare_hate_speech_dataset():
    df_dataset = _create_df_and_drop_columns(
        "data/original/hate-speech-dataset/annotations_metadata.csv",
        None,
        ["user_id", "subforum_id", "num_contexts"],
    )
    df_dataset = _filter_and_format_hate_speech(df_dataset)
    return df_dataset


def _filter_and_format_hate_speech(df):
    df.loc[df["label"] == "hate", "label"] = 0
    df.loc[df["label"] == "noHate", "label"] = 1
    df.rename(columns={"label": "class"}, inplace=True)
    content = []
    for i, row in df.iterrows():
        if row["class"] == "idk/skip" or row["class"] == "relation":
            df = df.drop(index=i)
            continue
        file = open(
            "data/original/hate-speech-dataset/all_files/{}.txt".format(row["file_id"]),
            encoding="utf-8",
            mode="r",
        )
        content.append(file.read())
        file.close()
    del df["file_id"]
    df["content"] = content
    return df