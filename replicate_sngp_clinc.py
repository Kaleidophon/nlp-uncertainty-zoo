"""
Script used to replicate the experiments of spectral-normalized Gaussian Process transformer by
`Liu et al. (2020) <https://arxiv.org/pdf/2006.10108.pdf>`_.
"""

# EXT
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.spectral_norm import SpectralNorm
from transformers import BertModel, BertTokenizer

# PROJECT
from src.spectral import SNGPModule
from datasets import load_dataset

# CONST
CLINC_DIR = "./data/processed/clinc"


class SNGPBert(nn.Module):
    def __int__(
        self,
        hidden_size: int,
        output_size: int,
        spectral_norm_upper_bound: float,
        scaling_coefficient: float,
        beta_length_scale: float,
    ):
        self.sngp_layer = SNGPModule(
            hidden_size, output_size, scaling_coefficient, beta_length_scale
        )
        self.bert = BertModel()

        # TODO: Manually implement spectral norm hook


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
