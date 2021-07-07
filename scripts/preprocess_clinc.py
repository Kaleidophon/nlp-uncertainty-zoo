"""
Preprocess the CLINC OOS dataset by writing splits into different files, tokenizing and
"""

# STD
import codecs
import json

# EXT
from transformers import BertTokenizer

# CONST
CLINC_PATHS = {
    "clinc": "../data/raw/clinc/data_full.json",
    "clinc_plus": "../data/raw/clinc/data_oos_plus.json",
}


if __name__ == "__main__":

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    for dataset, path in CLINC_PATHS.items():

        with open(path, "r") as file:
            data = json.load(file)

            for split, sentences in data.items():
                tokenized_sentences, labels = zip(
                    *[
                        (
                            tokenizer.tokenize(sentence),
                            label.replace("oos", "change_accent"),
                        )
                        for sentence, label in sentences
                    ]
                )

                # Filter sentences by length
                tokenized_sentences, labels = zip(
                    *list(
                        filter(
                            lambda tpl: len(tpl[0]) <= 32,
                            zip(tokenized_sentences, labels),
                        )
                    )
                )

                with codecs.open(
                    f"../data/processed/{dataset}/{split}.csv", "wb", "utf-8"
                ) as split_file:
                    for tokenized_sentence, label in zip(tokenized_sentences, labels):
                        split_file.write(f"{' '.join(tokenized_sentence)}\t{label}\n")
