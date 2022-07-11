# :robot::speech_balloon::question: nlp-uncertainty-zoo

This repository contains implementations of several models used for uncertainty estimation in Natural Language processing,
implemented in PyTorch. You can install the repository using pip:

    pip3 install nlp-uncertainty-zoo
    @TODO: Publish repo as package

If you are using the repository in your academic research, please cite the paper below:

    @TODO: Add citation

### Included models

The following models are implemented in the repository. They can all be imported by using `from nlp-uncertainty-zoo import <MODEL>`.
For transformer-based model, furthermore a version of a model is available that uses a pre-trained Bert from the HuggingFace hub.

| Name | Description | Implementation | Paper |
|---|---|---|---|
| LSTM | Vanilla LSTM | `LSTM` | [Hochreiter & Schmidhuber, 1997](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.676.4320&rep=rep1&type=pdf) |
| LSTM Ensemble | Ensemble of LSTMs | `LSTMEnsemble` | [Lakshminarayanan et al., 2017](https://proceedings.neurips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf) | 
| Bayesian LSTM | LSTM implementing Bayes-by-backprop [Blundell et al, 2015](http://proceedings.mlr.press/v37/blundell15.pdf) | `BayesianLSTM` | [Fortunato et al, 2017](https://arxiv.org/pdf/1704.02798.pdf) |
| ST-tau LSTM | LSTM modelling transitions of a finite-state-automaton | `STTauLSTM` | [Wang et al., 2021](https://openreview.net/pdf?id=9EKHN1jOlA) |
| Variational LSTM | LSTM with MC Dropout [Gal & Ghahramani, 2016a](http://proceedings.mlr.press/v48/gal16.pdf) | `VariationalLSTM` | [Gal & Ghahramani, 2016b](https://proceedings.neurips.cc/paper/2016/file/076a0c97d09cf1a0ec3e19c7f2529f2b-Paper.pdf) |
| DDU Transformer | Transformer with Gaussian Mixture Model fit to hidden features | `DDUTransformer`, `DDUBert` | [Mukhoti et al, 2021](https://arxiv.org/pdf/2102.11582.pdf) |
| Variational Transformer | Transformer with MC Dropout [Gal & Ghahramani, 2016a](http://proceedings.mlr.press/v48/gal16.pdf) | `VariationalTransformer`, `VariationalBert` | [Xiao et al., 2021](https://arxiv.org/pdf/2006.08344.pdf) |
| DPP Transformer | Transformer using determinantal point process dropout | `DPPTransformer`, `DPPBert` | [Shelmanov et al., 2021](https://aclanthology.org/2021.eacl-main.157) |
| SNGP Transformer | Spectrally-normalized transformer using a Gaussian Process output layer | `SNGPTransformer`, `SNGPBert` | [Liu et al., 2022](http://arxiv.org/abs/2205.00403) |

Contributions to include even more approaches are much appreciated!

### Usage

Each model comes in two versions, for instance `LSTMEnsemble` and `LSTMEnsembleModule`. The first one is supposed to be 
used as an out-of-the-box solution, encapsulating all training logic and convenience functions. These include fitting 
the model, prediction, getting the uncertainty for an input batch using a specific metric.

```python
model = LSTMEnsemble(**network_params, ensemble_size=10, is_sequence_classifer=False)
model.fit(train_split=train_dataloader)
model.get_logits(X)
model.get_predictions(X)
model.get_sequence_representation(X)
model.get_uncertainty(X)
model.get_uncertainty(X, metric_name="mutual_information")
```

In comparison, the `-Module` class is supposed to me more simple and bare-bones, only containing the core model logic. 
It is intended for research purposes, and for others who would like to embed the model into their own code base. While 
the model class (e.g. `LSTMEnsemble`) inherits from `Model` and would require to implement certain methods, any `Module` class
sticks closely to `torch.nn.Module`.

To check what arguments are required to initialize and use different models, check the documentation here (@TODO: Build documentation).

### Repository structure

The repository has the following structure:

* `model_configs`: Config files containing the parameters for all the models for a given dataset.
* `models`: All model implementations.
* `tests`: Unit tests. So far, only contains rudimentary tests to check that all output shapes are consistent between models.
* `utils`: Utility code (see below)
    * `utils/custom_types.py`: Custom types used in the repository for type annotations.
    * `utils/data.py`: Module containing data collators, and data builders - which build the dataloaders for a type of task and a specific dataset. Currently, language modelling, sequence labeling and sequence classification are supported.
    * `utils/metrics.py`: Implementations of uncertainty metrics.
    * `utils/samplers.py`: Dataset subsamplers for language modelling, sequence labelling and sequence classification.
    * `utils/task_eval.py`: Functions used to evaluate task performance.
    * `utils/uncertainty_eval.py`: Function used to evaluate uncertainty quality.
* `config.py`: Define available datasets, model and tasks.

