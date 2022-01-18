"""
Module to implement data reading and batching functionalities.
"""

# STD
from abc import ABC, abstractmethod
from functools import reduce
from typing import Dict, Optional, Any, List, Union

# EXT
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    PreTrainedTokenizerBase,
    DataCollatorForLanguageModeling,
    BertTokenizer,
)
from transformers.data.data_collator import BatchEncoding, _collate_batch


class ModifiedDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """
    Modified version of the DataCollatorForLanguageModelling. The only change introduced is to the __call__ function,
    where an offset between input_ids and labels for next token prediction language modelling in order to be consistent
    with the rest of the code base.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(
                examples,
                return_tensors="pt",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        else:
            batch = {
                "input_ids": _collate_batch(
                    examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
                )
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()

            # ### Custom Modification ###
            labels = labels[:, 1:]
            batch["input_ids"] = batch["input_ids"][:, :-1]
            # ### Custom Modifcation End ###

            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch


class DatasetBuilder(ABC):
    """
    Abstract dataset builder class used to create a variety of different dataset types, including sequence prediction,
    token prediction, next-token-prediction language modelling and masked language modelling.
    """

    def __init__(
        self,
        name: str,
        data_dir: str,
        splits: Dict[str, Any],
        type_: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        num_jobs: Optional[int] = 1,
    ):
        """
        Initialize a DatasetBuilder.

        Parameters
        ----------
        name: str
            Name of the final dataset.
        data_dir: str
            Directory in which data splits are located.
        splits: Dict[str, Any]
            Dictionary pointing to the files containing the training, validation and test split.
        type_: str
            String that further specifies the type of dataset being built (see LanguageModellingDatasetBuilder and
            ClassificationDatasetBuilder for more detail).
        tokenizer: PreTrainedTokenizerBase
            Pre-trained tokenizer.
        max_length: int
            Maximum sequence length.
        num_jobs: Optional[int]
            Number of jobs used to build the dataset (on CPU). Default is 1.
        """
        self.name = name
        self.data_dir = data_dir
        self.splits = splits
        self.type = type_
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_jobs = num_jobs

        # To be set after calling build()
        self.dataset = None
        self.dataloaders = None

    @abstractmethod
    def build(
        self, batch_size: int, **dataloader_kwargs: Dict[str, Any]
    ) -> Dict[str, DataLoader]:
        """
        Build a dataset.

        Parameters
        ----------
        batch_size: int
            The desired batch size.

        Returns
        -------
        Dict[str, DataLoader]
            Dictionary of DataLoaders for every given split.
        """
        pass


class LanguageModellingDatasetBuilder(DatasetBuilder):
    """
    DatasetBuilder for language modelling datasets. This includes "classic" language modelling (aka next token
    prediction) as well as masked language modelling.
    """

    def build(
        self, batch_size: int, **dataloader_kwargs: Dict[str, Any]
    ) -> Dict[str, DataLoader]:
        """
        Build a language modelling dataset.

        Parameters
        ----------
        batch_size: int
            The desired batch size.

        Returns
        -------
        Dict[str, DataLoader]
            Dictionary of DataLoaders for every given split.
        """
        assert self.type in [
            "mlm",
            "next_token_prediction",
        ], f"Invalid type '{self.type}' found, must be mlm or next_token_prediction."

        self.dataset = load_dataset("text", data_files=self.splits)

        # The following is basically copied from the corresponding HuggingFace tutorial: https://youtu.be/8PmhEIXhBvI
        self.dataset = self.dataset.map(
            lambda inst: self.tokenizer(
                inst["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            ),
            batched=True,
            num_proc=self.num_jobs,
        )
        self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        mlm_kwargs = (
            {"mlm": True, "mlm_probability": 0.15}
            if self.type == "mlm"
            else {"mlm": False}
        )

        # Note: Collater is a callable object used as the Pytorch DataLoader collate_fn arg
        collator = ModifiedDataCollatorForLanguageModeling(self.tokenizer, **mlm_kwargs)

        # Create the DataLoader for every split
        self.dataloaders = {
            split: DataLoader(
                self.dataset[split],
                batch_size=batch_size,
                collate_fn=collator,
                **dataloader_kwargs,
            )
            for split in self.splits
        }

        return self.dataloaders


class PennTreebankBuilder(LanguageModellingDatasetBuilder):
    """
    Dataset class for the Penn Treebank.
    """

    def __init__(self, data_dir: str, max_length: int, num_jobs: Optional[int] = 1):
        super().__init__(
            name="ptb",
            data_dir=data_dir,
            splits={
                "train": f"{data_dir}/ptb.train.txt",
                "valid": f"{data_dir}/ptb.valid.txt",
                "test": f"{data_dir}/ptb.test.txt",
            },
            type_="next_token_prediction",
            tokenizer=BertTokenizer.from_pretrained("bert-base-cased"),
            max_length=max_length,
            num_jobs=num_jobs,
        )


class ClassificationDatasetBuilder(DatasetBuilder):
    """
    DatasetBuilder for classification datasets. This includes sequence classification and token classification /
    sequence labelling.
    """

    def build(
        self, batch_size: int, **dataloader_kwargs: Dict[str, Any]
    ) -> Dict[str, DataLoader]:
        """
        Build a language modelling dataset.

        Parameters
        ----------
        batch_size: int
            The desired batch size.

        Returns
        -------
        Dict[str, DataLoader]
            Dictionary of DataLoaders for every given split.
        """
        assert self.type in [
            "sequence_classification",
            "token_classification",
        ], f"Invalid type '{self.type}' found, must be sequence_classification or token_classification."

        self.dataset = load_dataset(
            "csv",
            data_files=self.splits,
            delimiter="\t",
            column_names=["sentence", "label"],
        )

        # Extract all classes from data
        classes = None
        if self.type == "sequence_classification":
            # This one-liner goes through all the labels occuring in the different splits and adds them to a set
            classes = reduce(
                lambda x, y: set(x).union(y),
                [self.dataset[split]["label"] for split in self.splits],
            )

        # TODO: Modify and debug
        elif self.type == "token_classification":
            classes = reduce(
                lambda x, y: set(x).union(y),
                [self.dataset[split]["label"] for split in self.splits],
            )

        # Encode classes
        label_encoder = LabelEncoder()
        label_encoder.fit(list(classes))

        # Replace with classes with labels
        self.dataset = self.dataset.map(
            # TODO: The lambda function below likely needs to be changed for token_classification
            lambda inst: {"label": label_encoder.transform([inst["label"]])[0]},
            batched=False,
            with_indices=False,
            num_proc=self.num_jobs,
        )

        # The following is basically copied from the corresponding HuggingFace tutorial: https://youtu.be/8PmhEIXhBvI
        self.dataset = self.dataset.map(
            lambda inst: self.tokenizer(
                inst["sentence"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            ),
            batched=True,
            num_proc=self.num_jobs,
        )
        self.dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

        # Create the DataLoader for every split
        self.dataloaders = {
            split: DataLoader(
                self.dataset[split],
                batch_size=batch_size,
                **dataloader_kwargs,
            )
            for split in self.splits
        }

        # TODO: Debug
        for batch in self.dataloaders["train"]:
            ...

        return self.dataloaders


class ClincBuilder(ClassificationDatasetBuilder):
    """
    Dataset class for the CLINC OOS dataset.
    """

    def __init__(self, data_dir: str, max_length: int, num_jobs: Optional[int] = 1):
        super().__init__(
            name="clinc",
            data_dir=data_dir,
            splits={
                "train": f"{data_dir}/train.csv",
                "valid": f"{data_dir}/val.csv",
                "test": f"{data_dir}/test.csv",
                "oos_test": f"{data_dir}/oos_test.csv",
            },
            type_="sequence_classification",
            tokenizer=BertTokenizer.from_pretrained("bert-base-cased"),
            max_length=max_length,
            num_jobs=num_jobs,
        )


if __name__ == "__main__":
    # TODO: Debug
    # dataset = PennTreebankBuilder(
    #    data_dir="../data/processed/ptb", max_length=32, num_jobs=1
    # ).build(16)
    dataset = ClincBuilder(
        data_dir="../data/processed/clinc", max_length=32, num_jobs=1
    ).build(16)
    ...
