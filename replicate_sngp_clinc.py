"""
Script used to replicate the experiments of spectral-normalized Gaussian Process transformer by
`Liu et al. (2020) <https://arxiv.org/pdf/2006.10108.pdf>`_.
"""

# EXT
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


class SNGPBert(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        spectral_norm_upper_bound: float,
        ridge_factor: float,
        scaling_coefficient: float,
        beta_length_scale: float,
    ):
        super().__init__()

        # Model initialization
        self.sngp_layer = SNGPModule(
            hidden_size,
            output_size,
            ridge_factor,
            scaling_coefficient,
            beta_length_scale,
        )
        self.bert = BertModel.from_pretrained(BERT_MODEL)

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

    def forward(self, x: torch.LongTensor, attention_mask: torch.FloatTensor):
        pooler_output = self.bert.forward(x, attention_mask, return_dict=True)[
            "pooler_output"
        ]
        out = self.sngp_layer(pooler_output, update_sigma_hat_inv=self.last_epoch)

        return out

    def predict(self):
        ...  # TODO: Implement

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
    )

    # TODO: Init summary writer
    # TODO: Init knockknockbot
    # TODO: Move all tensors / model to correct device

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

    # TODO: Implement eval
