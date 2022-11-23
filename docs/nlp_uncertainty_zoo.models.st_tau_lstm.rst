ST-tau LSTM
===========

This model implement the ST-tau LSTM by `Wang et al. (2021) <https://openreview.net/pdf?id=9EKHN1jOlA>`_.
Intuitively, the ST-tau LSTM builds on earlier work that augments a LSTM architecture with a part that models the transitions
in a probabilistic finite-state automaton (with the ST refering to *St*ochastic FSAs).
At every step, a set of discrete state-transitions is sampled using the `Gumbel-softmax trick <https://sassafras13.github.io/GumbelSoftmax/>`_ and a temperature parameter **tau**,
which is a learnable parameter in the model in this approach.

Since the modelled FSA is *probabilistic*, we can quantify model uncertainty in the same way that we do for e.g. ensembles,
only that predictions do not originate from different parameter sets, but different paths through the FSA.

ST-tau LSTM Module Documentation
================================

.. automodule:: nlp_uncertainty_zoo.models.st_tau_lstm
   :members:
   :show-inheritance:
   :undoc-members: