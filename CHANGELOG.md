## 1.0.2

- Fix bug in DDUMixin that called an outdated function that was renamed in 1.0.0.

## 1.0.0

- Massive changes to repository structure, including restructuing of APIs and bug fixes. See the points below:
    - Demo notebook added, showcasing some common use cases. Take a look [here](https://github.com/Kaleidophon/nlp-uncertainty-zoo/blob/main/demo.ipynb) or [here](https://colab.research.google.com/drive/1-Pl5lvcnpbGL2ZXLGDDNqvJB7Ew8uIsS?usp=sharing).
    - API changes:
        - Instead of using a `model_params` dict when initializing a `Model` subclass, the necessary model arguments are spelled out 
    explicitly, similarly to the API of scikit-learn` models. Furthermore, reasonable default values are set for most model parameters.
        - The `Module` class implements additional functionalities:
            -  An `available_uncertainty_metrics` attribute gives some information about available uncertainty metrics, mapping from their names to their functions.
            - `get_sequence_representation()` was renamed to `get_sequence_representation_from_hidden()`. This was done to distinguish it from the new `get_sequence_classification()` function, which retrieves sequence representation directly from the inputs.
            - `get_hidden_representation()` defines how to obtain hidden representations from an input.
        - An explicit function called `compute_loss_weights()` was added to the `Model` class to allow to customize loss weights for unbalanced problems.
        - BERT-related models now have a `bert_class` argument, that allow the underlying model to be changed to a RoBERTa, DistilBert or similar model.'
        - Split `utils.uncertainty_eval` into  `utils.uncertainty_eval` and `utils.calibration_eval`.
        - There are now explicit functions to evaluate calibration and uncertainty properties on model, specified in `utils.uncertainty_eval.evaluate_uncertainty()` and `utils.calibration_eval.evaluate_calibration()`.
    - Bug fixes:
        - Changing batch size during inference won't produce shape errors for LSTM models anymore.


## 0.9.2

- Fix import issues concerning the `transformers` and `protobuf` packages mentioned in issues [#8](https://github.com/Kaleidophon/nlp-uncertainty-zoo/issues/8) and [#9](https://github.com/Kaleidophon/nlp-uncertainty-zoo/issues/9).
