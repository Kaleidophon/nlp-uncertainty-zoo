BERT
====

This module includes classes to enable compatibility with BERT models (`Devlin et al., 2018 <https://arxiv.org/pdf/1810.04805.pdf&usg=ALkJrhhzxlCL6yTht2BRmH9atgvKFxHsxQ>`_) that are stored in the `HuggingFace hub <https://huggingface.co/models>`_.
For instance, in the work this package was originally developed for (`Ulmer et al., 2022 <https://arxiv.org/pdf/2210.15452.pdf>`_), three different `BERTs` were used:

    * The original, English BERT by `Devlin et al. (2018) <https://arxiv.org/pdf/1810.04805.pdf&usg=ALkJrhhzxlCL6yTht2BRmH9atgvKFxHsxQ>`_ (`bert-base-uncased`).
    * The Danish BERT developed by `Hvingelby et al. (2020) <https://aclanthology.org/2020.lrec-1.565.pdf>`_ (`alexanderfalk/danbert-small-cased`).
    * The Finnish BERT provided by `Virtanen et al. (2019) <https://arxiv.org/pdf/1912.07076.pdf>`_ (`TurkuNLP/bert-base-finnish-cased-v1`).

The BERT model that is supposed to be used can be specified by using the `bert_name` argument for :py:meth:`nlp_uncertainty_zoo.models.bert.BertModule.__init__()`.

Documentation
=============

.. automodule:: nlp_uncertainty_zoo.models.bert
   :imported-members:
   :members:
   :show-inheritance:
   :undoc-members: