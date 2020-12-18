import pandas as pd


class SpecialCharacters:
    def __init__(self):
        self.list_of_special_characters = {
            "exclamation_mark": "!",
            "question_mark": "?",
            "full_stop_mark": ".",
        }

    def extract_features(self, df):
        """Extract number of special characters per data instance
        Parameters:
            df with the columns: class and content
        Return:
            passed df with new feature columns containing the count of the special character
        """
        for key in self.list_of_special_characters:
            df = self._count_number_of_special_characters(
                key, self.list_of_special_characters[key], df
            )
        return df

    def _count_number_of_special_characters(self, character_name, character, df):
        feature_name = "number_of_" + character_name
        df[feature_name] = df["content"].apply(
            lambda cell: self._count_character(cell, character)
        )
        return df

    def _count_character(self, sentence, character):
        return sentence.count(character)


if __name__ == "__main__":
    df_dataset = pd.read_csv("../../data/preprocessed/dataset.csv", index_col=0)

    special_characters = SpecialCharacters()
    df_with_extracted_features = special_characters.extract_features(df_dataset)
    print(df_with_extracted_features)
