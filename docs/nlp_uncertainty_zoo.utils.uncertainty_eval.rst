Uncertainty Eval
================

The quality of uncertainty estimates can be tricky to evaluate, since there are no gold labels like for a classification tasks.
For that reason, this module contains methods for exactly this purpose.

To measure the quality of general uncertainty estimates, we use the common evaluation method of defining a proxy OOD detection tasks,
where we quantify how well we can distinguish in- and out-of-distribution inputs based on uncertainty scores given by a model.
This is realized using the area under the receiver-operator-characteristic (:py:func:`nlp_uncertainty_zoo.utils.uncertainty_eval.aupr`) and
the area under precision-recall curve (:py:func:`nlp_uncertainty_zoo.utils.uncertainty_eval.auroc`).

New in `Ulmer et al. (2022) <https://arxiv.org/pdf/2210.15452.pdf>`_, it is also evaluated how indicative the uncertainty score is with the potential loss of a model.
For this reason, this module also implements the Kendall's tau correlation coefficient (:py:func:`nlp_uncertainty_zoo.utils.uncertainty_eval.kendalls_tau`).

Instances of the :py:class:`nlp_uncertainty_zoo.models.model.Model` class can be evaluated in a single function using all the above metrics with :py:func:`nlp_uncertainty_zoo.uncertainty_eval.evaluate_uncertainty`.

Uncertainty Eval Module Documentation
=====================================

.. automodule:: nlp_uncertainty_zoo.utils.uncertainty_eval
   :members:
   :show-inheritance:
   :undoc-members:
