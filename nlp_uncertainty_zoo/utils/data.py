"""
Module to implement data reading and batching functionalities.
"""

# STD
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import reduce
from typing import Dict, Optional, Any, List, Union, Type

# EXT
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    PreTrainedTokenizerBase,
    DataCollatorForLanguageModeling,
    BertTokenizer,
)
from transformers.data.data_collator import BatchEncoding, _collate_batch

# TYPES
# Either one set of keyword arguments per split or dictionary mapping split to split-specific keyword args
SamplerKwargs = Union[Dict[str, Any], Dict[str, Dict[str, Any]]]


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
            batch["attention_mask"] = batch["attention_mask"][:, :-1]
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
        sampler_class: Optional[Type] = None,
        sampler_kwargs: Optional[SamplerKwargs] = None,
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
        sampler_class: Optional[Sampler]
            Sampler class used to (sub-)sample the dataset.
        sampler_kwargs: Optional[SamplerKwargs]
            Keyword arguments to initialize the sampler.
        num_jobs: Optional[int]
            Number of jobs used to build the dataset (on CPU). Default is 1.
        """
        self.name = name
        self.data_dir = data_dir
        self.splits = splits
        self.type = type_
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sampler_class = sampler_class
        self.sampler_kwargs = sampler_kwargs
        self.num_jobs = num_jobs
        self.label_encoder = None

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
                sampler=self.sampler_class(
                    data_source=self.dataset[split],
                    **self.sampler_kwargs[split],
                )
                if self.sampler_class is not None and split in self.sampler_kwargs
                else None,
                **dataloader_kwargs,
            )
            for split in self.splits
        }

        return self.dataloaders


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
            column_names=["sentence", "labels"],
        )

        # Extract all classes from data
        classes, labeling_func = None, None
        if self.type == "sequence_classification":
            # This one-liner goes through all the labels occuring in the different splits and adds them to a set
            classes = reduce(
                lambda x, y: set(x).union(y),
                [self.dataset[split]["labels"] for split in self.splits],
            )
            labeling_func = lambda inst: {
                "labels": self.label_encoder.transform([inst["labels"]])[0]
            }

        elif self.type == "token_classification":
            classes = reduce(
                lambda x, y: set(x).union(y),
                [
                    labels.split(" ")
                    for split in self.splits
                    for labels in self.dataset[split]["labels"]
                ],
            )
            labeling_func = lambda inst: {
                "labels": self.label_encoder.transform(inst["labels"].split(" "))
            }

        # Encode classes
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(list(classes))

        # Replace with classes with labels
        self.dataset = self.dataset.map(
            labeling_func,
            batched=False,
            with_indices=False,
            num_proc=self.num_jobs,
        )

        # The following is basically copied from the corresponding HuggingFace tutorial: https://youtu.be/8PmhEIXhBvI
        if self.type == "sequence_classification":
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

        # For token classification, we have to adjust the token labels according to the subwords that the tokenizer
        # creates. Unfortunately, we have to do that manually and the code is quite ugly. Sorry!
        elif self.type == "token_classification":

            def create_inst_data(inst: Dict[str, Any]) -> Dict[str, Any]:
                # Dummy object, see explanations below
                class LabelledInfo:
                    val = False

                sentence, labels = inst["sentence"], inst["labels"]

                # Create data structure that maps the index of a character belonging to a token to the tokens label.
                # Also, create second structure mapping from character index to bool indicating whether token this
                # character belongs to was already labeled once - this way, splitting one labelled token into
                # multiple sub-word tokens later only labels the first ones and gives the ignore label -100 to the
                # other ones.
                char_idx2label = defaultdict(lambda: -100)
                char_idx2relabeled = defaultdict(lambda: LabelledInfo())
                char_idx = 0
                for token_idx, token in enumerate(sentence.split(" ")):
                    # Dirty trick: Assign the same INSTANCE of a bool value wrapped in an object to all char indices
                    # of a sub-word such that changing one of them changes all of them, since they all point to the
                    # same object in memory. I know this is ugly and confusing! I also don't like this!
                    # Don't you think that I also know that this is insane??
                    inf = LabelledInfo()
                    char_idx2relabeled.update(
                        {i: inf for i in range(char_idx + 1, char_idx + len(token) + 1)}
                    )

                    # Update the dict mapping from char index to label
                    char_idx2label.update(
                        {
                            # The min() expression is an ugly fix for cases in which a tokenizer butchers unusual
                            # expressions like an emoticon at the end of the sequence, suddenly creating a missing
                            # label. In this case, just repeat the last sequence label.
                            i: labels[min(token_idx, len(labels) - 1)]
                            for i in range(char_idx + 1, char_idx + len(token) + 1)
                        }
                    )

                    char_idx += len(token) + 1  # Account for whitespace

                inst_encoding = self.tokenizer(
                    sentence,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_offsets_mapping=True,
                )

                adjusted_labels = []  # New labels for sub-word tokens go here
                # Here, we have one single offset mapping per sub-word token, indicating which characters it span in
                # original sentence.
                for _, char_end in inst_encoding["offset_mapping"]:
                    # Check if one of the sub-word tokens belong to a word has been labelled already
                    if not char_idx2relabeled[char_end].val:
                        adjusted_labels.append(char_idx2label[char_end])
                        char_idx2relabeled[char_end].val = True

                    else:
                        adjusted_labels.append(-100)

                # Update the instance info
                return {**inst_encoding.data, "labels": adjusted_labels}

            # Finally map that shit function over the dataset
            self.dataset = self.dataset.map(
                create_inst_data,
                batched=False,
                num_proc=self.num_jobs,
            )

        self.dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )

        # Create the DataLoader for every split
        self.dataloaders = {
            split: DataLoader(
                self.dataset[split],
                batch_size=batch_size,
                sampler=self.sampler_class(
                    data_source=self.dataset[split],
                    **self.sampler_kwargs[split],
                )
                if self.sampler_class is not None and split in self.sampler_kwargs
                else None,
                **dataloader_kwargs,
            )
            for split in self.splits
        }

        return self.dataloaders