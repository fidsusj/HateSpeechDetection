""" This module prepares the two datasets """
import re

import pandas as pd
from path_helper import get_project_root


def prepare_and_merge_datasets(include_offensive_language=False):
    """ Prepares and merges the datasets """
    dataset_csv = open(
        str(get_project_root()) + "/data/preprocessed/dataset.csv",
        encoding="utf-8",
        mode="w",
    )
    dataset_copy = open(
        str(get_project_root()) + "/analysis/dataset.csv", encoding="utf-8", mode="w"
    )

    if include_offensive_language:
        df_dataset = _prepare_hate_speech_and_offensive_language(True)
    else:
        df_first_dataset = _prepare_hate_speech_and_offensive_language()
        df_second_dataset = _prepare_hate_speech_dataset()
        df_dataset = pd.concat([df_first_dataset, df_second_dataset], ignore_index=True)

    dataset_csv.write(df_dataset.to_csv())
    dataset_csv.close()

    dataset_copy.write(df_dataset.to_csv())
    dataset_copy.close()


# --- Hatespeech and Offensive Language ---


def _prepare_hate_speech_and_offensive_language(include_offensive_language=False):
    df_dataset = _create_df_and_drop_columns(
        str(get_project_root())
        + "/data/original/hate-speech-and-offensive-language/labeled_data.csv",
        0,
        ["count", "hate_speech", "offensive_language", "neither"],
    )

    if include_offensive_language:
        df_dataset.rename(columns={"tweet": "content"}, inplace=True)
    else:
        df_dataset = _filter_and_format_hate_speech_and_offensive_language(df_dataset)

    # df_dataset = _data_preparation(df_dataset)
    return df_dataset


def _create_df_and_drop_columns(path_to_csv, pd_index_col, list_columns_to_be_dropped):
    df = pd.read_csv(path_to_csv, index_col=pd_index_col)
    df.drop(list_columns_to_be_dropped, axis=1, inplace=True)
    return df


def _filter_and_format_hate_speech_and_offensive_language(df):
    df = df.drop(df[df["class"] == 1].index)
    df.loc[df["class"] == 2, "class"] = 1
    df.rename(columns={"tweet": "content"}, inplace=True)
    return df


def _data_preparation(df):
    for i, row in df.iterrows():
        if re.search(r"\"[^\"].*\"", row["content"]):
            df = df.drop(index=i)  # delete all tweet referencing tweets
        elif re.search(r"\"", row["content"]):
            df = df.drop(index=i)  # delete all citing tweets
    return df


# --- Hatespeech dataset ---


def _prepare_hate_speech_dataset():
    df_dataset = _create_df_and_drop_columns(
        str(get_project_root())
        + "/data/original/hate-speech-dataset/annotations_metadata.csv",
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
            str(get_project_root())
            + "/data/original/hate-speech-dataset/all_files/{}.txt".format(
                row["file_id"]
            ),
            encoding="utf-8",
            mode="r",
        )
        content.append(file.read())
        file.close()
    df["content"] = content

    df.drop(["file_id"], axis=1, inplace=True)
    return df
