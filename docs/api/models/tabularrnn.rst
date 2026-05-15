TabulaRNN
=========

Recurrent neural network for tabular data. TabulaRNN treats the feature vector
as a sequence of tokens and processes it with a recurrent cell. The cell type is
configurable: ``RNN``, ``LSTM``, ``GRU``, ``mLSTM`` (matrix LSTM), or
``sLSTM`` (scalar LSTM from the xLSTM family). This makes it a flexible
sequence model that spans classical to modern recurrent architectures.

When to Use
-----------

Best suited for datasets where feature ordering encodes meaningful structure —
for example, temporally ordered measurements stored as columns. Also a viable
alternative to Transformer-based models when memory efficiency is a priority.

Limitations
-----------

- Performance is sensitive to feature ordering; shuffling columns can
  significantly change results.
- May underperform Transformer architectures on unordered tabular data where
  positional bias is irrelevant.
- The mLSTM and sLSTM variants are newer and less empirically validated.

API Reference
-------------

.. currentmodule:: deeptab.models

.. autoclass:: TabulaRNNRegressor
   :members:
   :undoc-members:
   :noindex:

.. autoclass:: TabulaRNNClassifier
   :members:
   :undoc-members:
   :noindex:

.. autoclass:: TabulaRNNLSS
   :members:
   :undoc-members:
   :noindex:
