deeptab.metrics
===============

.. currentmodule:: deeptab.metrics

Base Class
----------

.. autoclass:: DeepTabMetric

Registry
--------

.. autodata:: METRIC_REGISTRY

.. autofunction:: get_default_metrics

.. autofunction:: get_default_metrics_dict

Regression Metrics
------------------

.. autoclass:: MeanSquaredError
   :members:
   :undoc-members:

.. autoclass:: RootMeanSquaredError
   :members:
   :undoc-members:

.. autoclass:: MeanAbsoluteError
   :members:
   :undoc-members:

.. autoclass:: R2Score
   :members:
   :undoc-members:

.. autoclass:: MeanAbsolutePercentageError
   :members:
   :undoc-members:

.. autoclass:: PinballLoss
   :members:
   :undoc-members:

Classification Metrics
-----------------------

.. autoclass:: Accuracy
   :members:
   :undoc-members:

.. autoclass:: F1Score
   :members:
   :undoc-members:

.. autoclass:: AUROC
   :members:
   :undoc-members:

.. autoclass:: AUPRC
   :members:
   :undoc-members:

.. autoclass:: LogLoss
   :members:
   :undoc-members:

.. autoclass:: BrierScore
   :members:
   :undoc-members:

.. autoclass:: ExpectedCalibrationError
   :members:
   :undoc-members:

Distributional / LSS Metrics
------------------------------

Proper Scoring Rules
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: NegativeLogLikelihood
   :members:
   :undoc-members:

.. autoclass:: LogScore
   :members:
   :undoc-members:

.. autoclass:: CRPS
   :members:
   :undoc-members:

.. autoclass:: IntervalScore
   :members:
   :undoc-members:

.. autoclass:: EnergyScore
   :members:
   :undoc-members:

Distribution-Specific Deviances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PoissonDeviance
   :members:
   :undoc-members:

.. autoclass:: GammaDeviance
   :members:
   :undoc-members:

.. autoclass:: TweedieDeviance
   :members:
   :undoc-members:

.. autoclass:: NegativeBinomialDeviance
   :members:
   :undoc-members:

.. autoclass:: BetaBrierScore
   :members:
   :undoc-members:

.. autoclass:: DirichletError
   :members:
   :undoc-members:

.. autoclass:: StudentTLoss
   :members:
   :undoc-members:

.. autoclass:: InverseGammaDeviance
   :members:
   :undoc-members:

.. autoclass:: LogNormalNLL
   :members:
   :undoc-members:

Calibration & Uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CoverageProbability
   :members:
   :undoc-members:

.. autoclass:: SharpnessScore
   :members:
   :undoc-members:

.. autoclass:: ProbabilityIntegralTransform
   :members:
   :undoc-members:
