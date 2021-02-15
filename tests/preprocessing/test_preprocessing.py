from unittest import TestCase

import pandas as pd
from preprocessing.corpus import build_corpus


class Test_Preprocessing(TestCase):
    raw_data = {
        "class": [1, 1, 1],
        "content": [
            "The U.S. Marshals Service said Barion A. Blake was killed about 12:30 p.m. at "
            "The Colonnade at Regency apartments in the 900 block of Monument Road .",
            "I 'll be out of town for a few weeks , and I don't plan on getting online again "
            "while I'm gone ... Maybe we can touch base then ! 88 !",
            "http://www.spamlaws.com/state/summary.html the site lists all 50 states I believe .",
        ],
    }

    def test_corpus(self):
        corpus = build_corpus(pd.DataFrame(data=self.raw_data))
        self.assertTrue(
            "p.m." in corpus.iloc[0]["tokens"]
        )  # check correct punctuation removal
        self.assertTrue(
            "will" in corpus.iloc[1]["tokens"]
        )  # check correct lemmatization
        self.assertTrue(
            "http://www.spamlaws.com/state/summary.html" in corpus.iloc[2]["tokens"]
        )
