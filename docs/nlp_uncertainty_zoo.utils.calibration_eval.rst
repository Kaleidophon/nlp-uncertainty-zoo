Calibration Eval
================

This module contains methods for exactly the purpose of evaluating the calibration properties of the model using calibration errors and prediction sets.

The module implements the expected calibration error `(Naeini et al., 2015) <https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9667/9958>`_, as well as the static calibration error and
the adaptive calibration error by `Nixon et al. (2019) <https://openaccess.thecvf.com/content_CVPRW_2019/papers/Uncertainty%20and%20Robustness%20in%20Deep%20Visual%20Learning/Nixon_Measuring_Calibration_in_Deep_Learning_CVPRW_2019_paper.pdf>`_, where the former is an extension to multiple classes, and the latter uses ranges instead of bins,
making sure that every range contains the same number of points.

.. warning::
    In `Ulmer et al. (2022) <https://arxiv.org/pdf/2210.15452.pdf>`_, we found that the SCE is not very informative for a relatively large number of classes (> 5).

Furthermore, we implement the evaluation of prediction sets by `Kompa et al. (2020) <http://arxiv.org/abs/2010.03039>`_: We determine the average width of prediction sets to reach 1 - alpha probability mass (:py:func:`nlp_uncertainty_zoo.utils.calibration_eval.coverage_width`),
and what percentage of prediction sets contain the correct class (:py:func:`nlp_uncertainty_zoo.utils.calibration_eval.coverage_percentage`).

Instances of the :py:class:`nlp_uncertainty_zoo.models.model.Model` class can be evaluated in a single function using all the above metrics with :py:func:`nlp_uncertainty_zoo.utils.calibration_eval.evaluate_calibration`.

Calibration Eval Module Documentation
=====================================

.. automodule:: nlp_uncertainty_zoo.utils.calibration_eval
   :members:
   :show-inheritance:
   :undoc-members:
