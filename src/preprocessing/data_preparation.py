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
    # Create dataframe and drop some columns
    df_dataset = pd.read_csv(
        "data/original/hate-speech-and-offensive-language/labeled_data.csv", index_col=0
    )
    del df_dataset["count"]
    del df_dataset["hate_speech"]
    del df_dataset["offensive_language"]
    del df_dataset["neither"]

    # Filter and format
    df_dataset = df_dataset.drop(
        df_dataset[df_dataset["class"] == 1].index
    )  # drop all offensive language documents
    df_dataset.loc[
        df_dataset["class"] == 2, "class"
    ] = 1  # set "neither" to class label 1
    df_dataset.rename(columns={"tweet": "content"}, inplace=True)

    # Data preparation
    for i, row in df_dataset.iterrows():
        df_dataset.at[i, "content"] = re.sub(
            "&.*?;", "", row["content"]
        )  # delete all emojies
        if re.search(r"\"[^\"].*\"", row["content"]):
            df_dataset = df_dataset.drop(index=i)  # delete all tweet referencing tweets
        elif re.search(r"\"", row["content"]):
            df_dataset = df_dataset.drop(index=i)  # delete all citing tweets

    return df_dataset


def _prepare_hate_speech_dataset():
    # Create dataframe and drop some columns
    df_dataset = pd.read_csv(
        "data/original/hate-speech-dataset/annotations_metadata.csv"
    )
    del df_dataset["user_id"]
    del df_dataset["subforum_id"]
    del df_dataset["num_contexts"]

    # Filter and format
    df_dataset.loc[df_dataset["label"] == "hate", "label"] = 0
    df_dataset.loc[df_dataset["label"] == "noHate", "label"] = 1
    df_dataset.rename(columns={"label": "class"}, inplace=True)

    content = []
    for i, row in df_dataset.iterrows():
        if row["class"] == "idk/skip" or row["class"] == "relation":
            df_dataset = df_dataset.drop(index=i)
            continue
        file = open(
            "data/original/hate-speech-dataset/all_files/{}.txt".format(row["file_id"]),
            encoding="utf-8",
            mode="r",
        )
        content.append(file.read())
        file.close()

    del df_dataset["file_id"]
    df_dataset["content"] = content

    # Data preparation

    return df_dataset
