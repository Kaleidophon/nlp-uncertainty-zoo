"""
Define Bert modules used in this project and make them consistent with the other models in the repository.
"""

# PROJECT
from nlp_uncertainty_zoo.models.model import Model, Module

# CONST
BERT_MODELS = {
    "english": "bert-base-uncased",
    "danish": "danbert-small-cased",
    "finnish": "bert-base-finnish-cased-v1",
    "swahili": "bert-base-multilingual-cased",
}

# TODO: Use different BERTs for different languages
# TODO: Ensure compatibility for (sequence) classification
# TODO: Write wrapper that implements all other functions defined by model / module class
# TODO: Write Model subclass with BERT batching
