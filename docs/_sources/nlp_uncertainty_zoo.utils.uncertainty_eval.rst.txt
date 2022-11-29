Uncertainty Eval
=================

Calibration and the quality of uncertainty estimates can be tricky to evaluate, since there are no gold labels like for a classification tasks.
For that reason, this module contains methods for exactly this purpose.

For calibration, the module implements the expected calibration error `(Naeini et al., 2015) <https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9667/9958>`_, as well as the static calibration error and
the adaptive calibration error by `Nixon et al. (2019) <https://openaccess.thecvf.com/content_CVPRW_2019/papers/Uncertainty%20and%20Robustness%20in%20Deep%20Visual%20Learning/Nixon_Measuring_Calibration_in_Deep_Learning_CVPRW_2019_paper.pdf>`_, where the former is an extension to multiple classes, and the latter uses ranges instead of bins,
making sure that every range contains the same number of points.

.. warning::
    In `Ulmer et al. (2022) <https://arxiv.org/pdf/2210.15452.pdf>`_, we found that the SCE is not very informative for a relatively large number of classes (> 5).

Furthermore, we implement the evaluation of prediction sets by `Kompa et al. (2020) <http://arxiv.org/abs/2010.03039>`_: We determine the average width of prediction sets to reach 1 - alpha probability mass (:py:func:`nlp_uncertainty_zoo.utils.uncertainty_eval.coverage_width`),
and what percentage of prediction sets contain the correct class (:py:func:`nlp_uncertainty_zoo.utils.uncertainty_eval.coverage_percentage`).

To measure the quality of general uncertainty estimates, we use the common evaluation method of defining a proxy OOD detection tasks,
where we quantify how well we can distinguish in- and out-of-distribution inputs based on uncertainty scores given by a model.
This is realized using the area under the receiver-operator-characteristic (:py:func:`nlp_uncertainty_zoo.utils.uncertainty_eval.aupr`) and
the area under precision-recall curve (:py:func:`nlp_uncertainty_zoo.utils.uncertainty_eval.auroc`).

New in `Ulmer et al. (2022) <https://arxiv.org/pdf/2210.15452.pdf>`_, it is also evaluated how indicative the uncertainty score is with the potential loss of a model.
For this reason, this module also implements the Kendall's tau correlation coefficient (:py:func:`nlp_uncertainty_zoo.utils.uncertainty_eval.kendalls_tau`).

Uncertainty Eval Module Documentation
=====================================

.. automodule:: nlp_uncertainty_zoo.utils.uncertainty_eval
   :members:
   :show-inheritance:
   :undoc-members:
