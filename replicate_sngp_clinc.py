"""
Script used to replicate the experiments of spectral-normalized Gaussian Process transformer by
`Liu et al. (2020) <https://arxiv.org/pdf/2006.10108.pdf>`_.
"""

# STD
import argparse
from datetime import datetime
import json
import os
from typing import Optional

# EXT
from codecarbon import OfflineEmissionsTracker
from knockknock import telegram_sender
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.spectral_norm import SpectralNorm
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup

# PROJECT
from src.spectral import SNGPModule
from src.types import Device
from datasets import load_dataset
from secret import COUNTRY_CODE, TELEGRAM_CHAT_ID, TELEGRAM_API_TOKEN

# CONST
CLINC_DIR = "./data/processed/clinc"
BERT_MODEL = "bert-base-uncased"
SEED = 123
EMISSION_DIR = "./emissions"

# HYPERPARAMETERS
HIDDEN_SIZE = 768
OUTPUT_SIZE = 151
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


def run_replication(num_runs: int, device: Device):
    """
    Run replication of CLINC experiments of `Liu et al. (2020) <https://arxiv.org/pdf/2006.10108.pdf>`_.

    Parameters
    ----------
    num_runs: int
        Number of random seeds that should be tried.
    device: Device
        Device the replication is performed on.

    Returns
    -------
    str
        String with results for knockknock.
    """
    accuracies, ood_aurocs = [], []

    for _ in range(num_runs):
        summary_writer = SummaryWriter()

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
            dl = DataLoader(dataset["train"], batch_size=BATCH_SIZE)

            for batch_num, batch in enumerate(dl):
                global_batch_num = epoch * len(dl) + batch_num

                # During the last epochs, update sigma_hat_inv matrix
                sngp_bert.last_epoch = epoch == EPOCHS - 1

                # Forward pass
                attention_mask, input_ids, labels = (
                    batch["attention_mask"],
                    batch["input_ids"],
                    batch["y"],
                )
                attention_mask, input_ids, labels = (
                    attention_mask.to(device),
                    input_ids.to(device),
                    labels.to(device),
                )

                out = sngp_bert(input_ids, attention_mask)
                del input_ids, attention_mask  # Desperately try to save memory
                loss = loss_func(out, labels)

                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Spectral normalization
                sngp_bert.spectral_normalization()

                # Invert sigma matrix during last epoch
                if epoch == EPOCHS - 1:
                    sngp_bert.sngp_layer.invert_sigma_hat()

                # Save training stats
                summary_writer.add_scalar(
                    "Batch train loss", loss.cpu().detach(), global_batch_num
                )
                summary_writer.add_scalar(
                    "Batch learning rate",
                    scheduler.get_last_lr()[0],
                    global_batch_num,
                )

        # ### Eval ###
        uncertainties_id, uncertainties_ood = [], []

        # Evaluate accuracy on test set
        with torch.no_grad():
            dl_test = DataLoader(dataset["test"], batch_size=BATCH_SIZE)
            total, correct = 0, 0

            for batch in dl_test:
                attention_mask, input_ids, labels = (
                    batch["attention_mask"],
                    batch["input_ids"],
                    batch["y"],
                )
                attention_mask, input_ids, labels = (
                    attention_mask.to(device),
                    input_ids.to(device),
                    labels.to(device),
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
            accuracy = accuracy.cpu().item()

            summary_writer.add_scalar("Accuracy", accuracy)
            accuracies.append(accuracy)

        # Evaluate OOD detection performance
        with torch.no_grad():
            dl_ood = DataLoader(dataset["oos_test"], batch_size=BATCH_SIZE)

            for batch in dl_ood:
                attention_mask, input_ids, labels = (
                    batch["attention_mask"],
                    batch["input_ids"],
                    batch["y"],
                )
                attention_mask, input_ids, labels = (
                    attention_mask.to(device),
                    input_ids.to(device),
                    labels.to(device),
                )

                # Get uncertainties for ID samples
                uncertainties = sngp_bert.get_uncertainty(input_ids, attention_mask)
                uncertainties = uncertainties.cpu().detach()
                uncertainties_ood.append(uncertainties)

        # Eval uncertainties using AUROC
        uncertainties_id = torch.cat(uncertainties_id, dim=0)
        uncertainties_ood = torch.cat(uncertainties_ood, dim=0)
        # Create "labels": 1 for ID, 0 for OOD
        ood_labels = [0] * uncertainties_id.shape[0] + [1] * uncertainties_ood.shape[0]
        uncertainties = np.concatenate(
            [
                uncertainties_id.cpu().detach().numpy(),
                uncertainties_ood.cpu().detach().numpy(),
            ],
            axis=0,
        )
        ood_auroc = roc_auc_score(ood_labels, uncertainties)
        summary_writer.add_scalar("ROC-AUC", ood_auroc)
        ood_aurocs.append(ood_auroc)

        # Add statistics to run
        summary_writer.add_hparams(
            hparam_dict={
                "hidden_size": HIDDEN_SIZE,
                "output_size": OUTPUT_SIZE,
                "batch_size": BATCH_SIZE,
                "spectral_norm_upperbound": SPECTRAL_NORM_UPPER_BOUND,
                "ridge_factor": RIDGE_FACTOR,
                "scaling_coefficient": SCALING_COEFFICIENT,
                "beta_length_scale": BETA_LENGTH_SCALE,
                "weight_decay": WEIGHT_DECAY,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "warmup_prop": WARMUP_PROP,
                "num_predictions": NUM_PREDICTIONS,
            },
            metric_dict={
                "accuracy": np.mean(accuracies),
                "ood_auc_roc": np.mean(ood_aurocs),
            },
        )
        # Reset for potential next run
        summary_writer.close()

    return json.dumps(
        {
            "dataset": "CLINC",
            "runs": num_runs,
            "accuracy": f"{np.mean(accuracies):.2f} ±{np.std(accuracies):.2f}",
            "ood_auc_roc": f"{np.mean(ood_aurocs):.2f} ±{np.std(ood_aurocs):.2f}",
        },
        indent=4,
        ensure_ascii=False,
    )


class SNGPBert(nn.Module):
    """
    Definition of a BERT model with a custom SNGP output layer.
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        spectral_norm_upper_bound: float,
        ridge_factor: float,
        scaling_coefficient: float,
        beta_length_scale: float,
        num_predictions: int,
        device: Device,
    ):
        """
        Initialize a SNGP-Bert.

        Parameters
        ----------
        hidden_size: int
            Hidden size of last Bert layer.
        output_size: int
            Size of output layer, so number of classes.
        spectral_norm_upper_bound: float
            Set a limit when weight matrices will be spectrally normalized if their lambda parameter surpasses it.
        ridge_factor: float
            Factor that identity sigma hat matrices of the SNGP layer are multiplied by.
        scaling_coefficient: float
            Momentum factor that is used when updating the sigma hat matrix of the SNGP layer during the last training
            epoch.
        beta_length_scale: float
            Factor for the variance parameter of the normal distribution all beta parameters of the SNGP layer are
            initialized from.
        num_predictions: int
            Number of predictions sampled from the GP in the SNGP layer to come to the final prediction.
        device: Device
            Device the replication is performed on.
        """
        super().__init__()
        self.device = device

        # Model initialization
        self.sngp_layer = SNGPModule(
            hidden_size,
            output_size,
            ridge_factor,
            scaling_coefficient,
            beta_length_scale,
            num_predictions,
            device,
        )
        self.bert = BertModel.from_pretrained(BERT_MODEL).to(device)
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
        """
        Forward pass of the model, used during training.

        Parameters
        ----------
        x: torch.LongTensor
            Input to the model, containing all token IDs of the current batch.
        attention_mask: torch.FloatTensor
            Attention mask for Bert for the current batch.

        Returns
        -------
        torch.FloatTensor
            Logits for the sequences of the current batch.
        """
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
        """
        Make predictions for data points.

        Parameters
        ----------
        x: torch.LongTensor
            Input to the model, containing all token IDs of the current batch.
        attention_mask: torch.FloatTensor
            Attention mask for Bert for the current batch.
        num_predictions: int
            Number of predictions sampled from the GP in the SNGP layer to come to the final prediction.

        Returns
        -------
        torch.FloatTensor
            Class probabilities for the sequences of the current batch.
        """
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
        """
        Get uncertainty scores for the current batch, using the Dempster-Shafer metric.

        Parameters
        ----------
        x: torch.LongTensor
            Input to the model, containing all token IDs of the current batch.
        attention_mask: torch.FloatTensor
            Attention mask for Bert for the current batch.
        num_predictions: int
            Number of predictions sampled from the GP in the SNGP layer to come to the final prediction.

        Returns
        -------
        torch.FloatTensor
            Uncertainty scores for the current batch.
        """
        if num_predictions is None:
            num_predictions = self.num_predictions

        pooler_output = self.bert.forward(x, attention_mask, return_dict=True)[
            "pooler_output"
        ]
        uncertainties = self.sngp_layer.dempster_shafer(pooler_output, num_predictions)

        return uncertainties

    def spectral_normalization(self):
        """
        Apply spectral normalization to the Bert pooling layer, but only when lambda exceeds spectral_norm_upper_bound.
        """
        # For Bert, only apply to pooler layer following Liu et al. (2020)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--emission-dir", type=str, default=EMISSION_DIR)
    parser.add_argument("--track-emissions", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--knock", action="store_true", default=False)
    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
        device=args.device,
    ).to(args.device)

    # Init emission tracking, summary writer, etc.
    if args.track_emissions:
        timestamp = str(datetime.now().strftime("%d-%m-%Y (%H:%M:%S)"))
        emissions_path = os.path.join(args.emission_dir, timestamp)
        os.mkdir(emissions_path)
        tracker = OfflineEmissionsTracker(
            project_name="nlp-uncertainty-zoo-experiments",
            country_iso_code=COUNTRY_CODE,
            output_dir=emissions_path,
        )
        tracker.start()

        # Apply decorator
        if args.knock:
            run_replication = telegram_sender(
                token=TELEGRAM_API_TOKEN, chat_id=TELEGRAM_CHAT_ID
            )(run_replication)

    try:
        run_replication(args.runs, args.device)

    except Exception as e:
        # Save data from emission tracker in any case
        if tracker is not None:
            tracker.stop()

        raise e
