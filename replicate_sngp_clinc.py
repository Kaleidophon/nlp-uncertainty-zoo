"""
Script used to replicate the experiments of spectral-normalized Gaussian Process transformer by
`Liu et al. (2020) <https://arxiv.org/pdf/2006.10108.pdf>`_.
"""

# STD
from typing import Optional

# EXT
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.spectral_norm import SpectralNorm
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup

# PROJECT
from src.spectral import SNGPModule
from datasets import load_dataset

# CONST
CLINC_DIR = "./data/processed/clinc"
BERT_MODEL = "bert-base-uncased"

# HYPERPARAMETERS
HIDDEN_SIZE = 768
OUTPUT_SIZE = 150
BATCH_SIZE = 32
SPECTRAL_NORM_UPPER_BOUND = 0.95
RIDGE_FACTOR = 0.001
SCALING_COEFFICIENT = 0.999
BETA_LENGTH_SCALE = 2
WEIGHT_DECAY = 0
EPOCHS = 40
LEARNING_RATE = 5e-5
WARMUP_PROP = 0.1
NUM_PREDICTIONS = 10


class SNGPBert(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        spectral_norm_upper_bound: float,
        ridge_factor: float,
        scaling_coefficient: float,
        beta_length_scale: float,
        num_predictions: int,
    ):
        super().__init__()

        # Model initialization
        self.sngp_layer = SNGPModule(
            hidden_size,
            output_size,
            ridge_factor,
            scaling_coefficient,
            beta_length_scale,
            num_predictions,
        )
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        self.output_size = output_size

        # Spectral norm initialization
        self.spectral_norm_upper_bound = spectral_norm_upper_bound
        self.spectral_norm = SpectralNorm.apply(
            self.bert.pooler.dense,
            name="weight",
            n_power_iterations=1,
            dim=0,
            eps=1e-12,
        )

        # Misc.
        self.last_epoch = False
        self.num_predictions = num_predictions

    def forward(self, x: torch.LongTensor, attention_mask: torch.FloatTensor):
        pooler_output = self.bert.forward(x, attention_mask, return_dict=True)[
            "pooler_output"
        ]
        out = self.sngp_layer(pooler_output, update_sigma_hat_inv=self.last_epoch)

        return out

    def predict(
        self,
        x: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        num_predictions: Optional[int] = None,
    ):

        if num_predictions is None:
            num_predictions = self.num_predictions

        pooler_output = self.bert.forward(x, attention_mask, return_dict=True)[
            "pooler_output"
        ]
        out = self.sngp_layer.predict(pooler_output, num_predictions=num_predictions)

        return out

    def get_uncertainty(
        self,
        x: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        num_predictions: Optional[int] = None,
    ):
        if num_predictions is None:
            num_predictions = self.num_predictions

        pooler_output = self.bert.forward(x, attention_mask, return_dict=True)[
            "pooler_output"
        ]
        uncertainties = self.sngp_layer.dempster_shafer(pooler_output, num_predictions)

        return uncertainties

    def spectral_normalization(self):
        # For BERT, only apply to pooler layer following Liu et al. (2020)
        pooler = self.bert.pooler.dense
        old_weight = pooler.weight.clone()
        normalized_weight = self.spectral_norm.compute_weight(
            pooler, do_power_iteration=True
        )
        u, v = pooler.weight_u.unsqueeze(1), pooler.weight_v.unsqueeze(1)
        lambda_ = u.T @ pooler.weight @ v  # Compute spectral norm for weight matrix

        if lambda_ > self.spectral_norm_upper_bound:
            self.bert.pooler.dense.weight = (
                self.spectral_norm_upper_bound * normalized_weight
            )

        else:
            self.bert.pooler.dense_weight = old_weight


if __name__ == "__main__":

    # Load dataset
    dataset = load_dataset(
        "csv",
        data_files={
            "train": f"{CLINC_DIR}/train.csv",
            "valid": f"{CLINC_DIR}/val.csv",
            "test": f"{CLINC_DIR}/test.csv",
            "oos_test": f"{CLINC_DIR}/oos_test.csv",
        },
        delimiter="\t",
        column_names=["sentence", "label"],
    )

    # Encode labels
    classes = (
        dataset["train"]["label"]
        + dataset["valid"]["label"]
        + dataset["test"]["label"]
        + ["oos"]
    )
    label_encoder = LabelEncoder()
    label_encoder.fit(classes)
    dataset = dataset.map(
        lambda e: {"y": label_encoder.transform([e["label"]])[0]},
        batched=False,
        with_indices=False,
    )

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    dataset = dataset.map(
        lambda e: tokenizer(e["sentence"], truncation=True, padding="max_length"),
        batched=True,
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "y"])

    # Init SNGP-BERT
    sngp_bert = SNGPBert(
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        spectral_norm_upper_bound=SPECTRAL_NORM_UPPER_BOUND,
        ridge_factor=RIDGE_FACTOR,
        scaling_coefficient=SCALING_COEFFICIENT,
        beta_length_scale=BETA_LENGTH_SCALE,
        num_predictions=NUM_PREDICTIONS,
    )

    # TODO: Init summary writer
    # TODO: Init knockknockbot
    # TODO: Move all tensors / model to correct device
    # TODO: Implement training over multiple seeds

    # ### Training ###

    # Init optimizer, loss
    steps_per_epoch = len(dataset["train"])
    optimizer = optim.Adam(
        sngp_bert.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=steps_per_epoch * EPOCHS * WARMUP_PROP,
        num_training_steps=steps_per_epoch * EPOCHS,
    )
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        dl = DataLoader(dataset["train"], batch_size=32)

        for batch in dl:
            # During the last epochs, update sigma_hat_inv matrix
            sngp_bert.last_epoch = epoch == EPOCHS - 1

            # Forward pass
            attention_mask, input_ids, labels = (
                batch["attention_mask"],
                batch["input_ids"],
                batch["y"],
            )

            out = sngp_bert(input_ids, attention_mask)
            loss = loss_func(out, labels)
            print("Loss: ", loss)

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Spectral normalization
            sngp_bert.spectral_normalization()

            if epoch == EPOCHS - 1:
                sngp_bert.sngp_layer.invert_sigma_hat()

    # ### Eval ###
    uncertainties_id, uncertainties_ood = [], []

    # Evaluate accuracy on test set
    with torch.no_grad():
        dl_test = DataLoader(dataset["test"], batch_size=32)
        total, correct = 0, 0

        for batch in dl_test:
            attention_mask, input_ids, labels = (
                batch["attention_mask"],
                batch["input_ids"],
                batch["y"],
            )

            # Get predictions for accuracy
            out = sngp_bert.predict(input_ids, attention_mask)
            preds = torch.argmax(out, dim=-1)
            total += preds.shape[0]
            correct = (preds == labels).long().sum()

            # Get uncertainties for ID samples
            uncertainties = sngp_bert.get_uncertainty(input_ids, attention_mask)
            uncertainties_id.append(uncertainties)

        accuracy = correct / total

    # Evaluate OOD detection performance
    with torch.no_grad():
        dl_ood = DataLoader(dataset["oos_test"], batch_size=32)

        for batch in dl_ood:
            attention_mask, input_ids, labels = (
                batch["attention_mask"],
                batch["input_ids"],
                batch["y"],
            )

            # Get uncertainties for ID samples
            uncertainties = sngp_bert.get_uncertainty(input_ids, attention_mask)
            uncertainties_ood.append(uncertainties)

    # Eval uncertainties using AUROC
    uncertainties_id = torch.cat(uncertainties_id, dim=0)
    uncertainties_ood = torch.cat(uncertainties_ood, dim=0)
    # Create "labels": 1 for ID, 0 for OOD
    ood_labels = [0] * uncertainties_id.shape[0] + [1] * uncertainties_ood.shape[0]
    uncertainties = np.concatenate(
        [uncertainties_id.numpy(), uncertainties_ood.numpy()], axis=0
    )
    ood_auroc = roc_auc_score(ood_labels, uncertainties)
