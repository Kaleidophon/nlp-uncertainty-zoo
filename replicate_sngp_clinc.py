"""
Script used to replicate the experiments of spectral-normalized Gaussian Process transformer by
`Liu et al. (2020) <https://arxiv.org/pdf/2006.10108.pdf>`_.
"""

# EXT
import torch.nn as nn
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
        "text",
        data_files={
            "train": f"{CLINC_DIR}/train.txt",
            "valid": f"{CLINC_DIR}/val.txt",
            "test": f"{CLINC_DIR}/test.txt",
        },
    )

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
