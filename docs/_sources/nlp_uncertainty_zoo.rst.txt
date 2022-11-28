nlp_uncertainty_zoo
===================

This repository implements several models and uncertainty metrics used to quantify the uncertainty and confidence for
Natural Language Processing applications. As of now, it is focused on sequence labelling and sequence classification
(with hopefully more NLP tasks coming in the future!).

For more background info and information and usage, please refer to the `README` on the `landing page <http://dennisulmer.eu/nlp-uncertainty-zoo/>`_.
On the highest level, the repository comprises multiple packages, including

    * :py:mod:`nlp_uncertainty_zoo.models`: All model implemmentations.
    * :py:mod:`nlp_uncertainty_zoo.utils`: Miscellaneous modules implementing uncertainty metrics, evaluation logic for task performance as well as calibration and uncertainty qualitaty evaluation methods. Also includes code for data management.

Please refer to the resource above for more information.
