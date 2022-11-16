nlp_uncertainty_zoo.models
=============

This module includes abstractions for the model implementation in this package, defining the basic signature
of the `__init__()` method and some mandatory methods that have to be implemented by subclasses.

The idea is to provide all implemented models in two versions to accommodate different users:

    * :py:class:`nlp_uncertainty_zoo.models.model.Module` only defines the basic model logic, but no training loop etc. This is ideal for research projects where you would like to integrate the model into your own repository structure or tinkering.
    * :py:class:`nlp_uncertainty_zoo.models.model.Model` defines and out-of-the-box solution for direct application. The model can simply be trained calling :py:meth:`nlp_uncertainty_zoo.models.model.Model.fit()` with some training dataloader.

The `Module` class
==================

The :py:class:`nlp_uncertainty_zoo.models.model.Module` class is supposed to mirror PyTorch's `nn.Module<https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_ class, only including the bare-bones logic of the
model. By default, input parameters include the number of layers, as well as the sizes for the model vocabulary, embeddings
(`input_size`), hidden activations and number of classes (`output_size`).

One also needs to specify the PyTorch `device` (like "cpu" or "cuda") and whether the model is used for sequence classification
(`is_sequence_classifier=True`) or sequence labelling (`is_sequence_classifier=False`). Other kind of tasks like regression,
parsing or generation are not supported at the moment.

The following methods have to be implemented by every subclass of `Module`:

    * :py:meth:`nlp_uncertainty_zoo.models.model.Module.get_logits`: Return the logits for a given input, which come in the form of a `torch.FloatTensor` with dimensions `batch_size x sequence_length x output_size` for models with only a single prediction or `batch_size x num_predictions x sequence_length x output_size` for models with multiple predictions, such as MC Dropout or ensembles. If the model is a sequence classifier, `sequence_length` will be 1.
    * :py:meth:`nlp_uncertainty_zoo.models.model.Module.predict`: Same as :py:meth:`nlp_uncertainty_zoo.models.model.Module.get_logits`, except that values on the last axis are actual probabilities summing up to 1.
    * :py:meth:`nlp_uncertainty_zoo.models.model.Module.get_sequence_representation()`: Returns the representation of a sequence of a certain model as a `torch.FloatTensor` of size `batch_size x hidden_size`. For :py:mod:`nlp_uncertainty_zoo.models.bert` and :py:mod:`nlp_uncertainty_zoo.models.transformer` models, the sequence representation is obtained by using the top-layer hidden activations of the first time step (often corresponding to the `[CLS]` token) after an additional pooler layer, or the last step hidden activations of the last layer of the unidirectional :py:mod:`nlp_uncertainty_zoo.models.lstm` classes.
    * :py:meth:`nlp_uncertainty_zoo.models.model.Module.get_uncertainty()`: Return the uncertainty estimates for an input batch, with the return tensor possessing the same shape as with :py:meth:`nlp_uncertainty_zoo.models.model.Module.get_logits` or :py:meth:`nlp_uncertainty_zoo.models.model.Module.predict. If no value is specified for `metric_name`, the metric stored in the attribute `default_uncertainty_metric` is used (which usually refers to predictive entropy. If another metric should be used, one of the names in the keys of the attributes `single_prediction_uncertainty_metrics` or `multi_prediction_uncertainty_metrics`.

The `Model` class
=================

The :py:class:`nlp_uncertainty_zoo.models.model.Model` class is aimed as a complete drop-in solution for anyone who does not want to write training logic and similar aspects.
As input parameters, :py:class:`nlp_uncertainty_zoo.models.model.Model` expected the name of the model as a string, a reference to a class, the model parameters
as a dictionary of keyword arguments that is passed to the :py:meth:`nlp_uncertainty_zoo.models.model.Module.__init__()` function of the given :py:class:`nlp_uncertainty_zoo.models.model.Module` subclass.
The `model_dir` is an optional argument that specified the path to which the model is saved during training.

The user mainly interacts with the :py:class:`nlp_uncertainty_zoo.models.model.Model` class using the :py:meth:`nlp_uncertainty_zoo.models.model.Model.fit()`, :py:meth:`nlp_uncertainty_zoo.models.model.Model.predict()` and :py:meth:`nlp_uncertainty_zoo.models.model.Model.get_uncertainty()` functions,
where the latter two mirror the function implementations in `Module`. :py:meth:`nlp_uncertainty_zoo.models.model.Model.fit()` expects a training and validation set
`torch.utils.data.DataLoader<https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_ instances.

Documentation
=============

.. automodule:: nlp_uncertainty_zoo.models
   :imported-members:
   :members:
   :show-inheritance:
   :undoc-members: