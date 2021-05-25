"""
Script used to replicate the experiments of spectral-normalized Gaussian Process transformer by
`Liu et al. (2020) <https://arxiv.org/pdf/2006.10108.pdf>`_.
"""

# EXT
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer

# PROJECT
from datasets import load_dataset

# CONST
CLINC_DIR = "./data/processed/clinc"


class SNGPBert(nn.Module):
    ...


if __name__ == "__main__":

    # Load dataset
    dataset = load_dataset(
        "csv",
        data_files={
            "train": f"{CLINC_DIR}/train.csv",
            "valid": f"{CLINC_DIR}/val.csv",
            "test": f"{CLINC_DIR}/test.csv",
        },
        delimiter="\t",
        column_names=["sentence", "label"],
    )

    # Encode labels
    classes = (
        dataset["train"]["label"] + dataset["valid"]["label"] + dataset["test"]["label"]
    )
    label_encoder = LabelEncoder()
    label_encoder.fit(classes)
    dataset = dataset.map(
        lambda e: {"y": label_encoder.transform([e["label"]])[0]},
        batched=False,
        with_indices=False,
    )

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = dataset.map(
        lambda e: tokenizer(e["sentence"], truncation=True, padding="max_length"),
        batched=True,
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "y"])
    dl = DataLoader(dataset["train"], batch_size=32)

    # Init BERT

    for batch in dl:
        ...
