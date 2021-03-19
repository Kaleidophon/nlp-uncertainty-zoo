"""
Implement a simple mixin that allows for inference using MC Dropout `Gal & Ghrahramani (2016a) `
<http://proceedings.mlr.press/v48/gal16.pdf> and corresponding subclasses of the LSTM and transformer, creating two
models:

* Variational LSTM `(Gal & Ghrahramani, 2016b) <https://arxiv.org/pdf/1512.05287.pdf>`
* Variational Transformer `(Xiao et al., 2021) <https://arxiv.org/pdf/2006.08344.pdf>`
"""

# PROJECT
from src.module import Module


class MCDropoutMixin:
    ...  # TODO


class VariationalLSTM(MCDropoutMixin):
    ...  # TODO


class VariationalLSTMModule(Module):
    ...  # TODO


class VariationalTransformer(MCDropoutMixin):
    ...  # TODO


class VariationalTransformerModule(Module):
    ...  # TODO
