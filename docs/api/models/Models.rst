deeptab.models
==============

Complete API reference for all DeepTab models. For usage examples and configuration guidance,
see :doc:`../../model_zoo/index`.

State Space Models
------------------

Mambular
~~~~~~~~

.. autoclass:: deeptab.models.MambularClassifier
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.MambularRegressor
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.MambularLSS
    :members:
    :inherited-members:

MambaTab
~~~~~~~~

.. autoclass:: deeptab.models.MambaTabClassifier
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.MambaTabRegressor
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.MambaTabLSS
    :members:
    :inherited-members:

MambAttention
~~~~~~~~~~~~~

.. autoclass:: deeptab.models.MambAttentionClassifier
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.MambAttentionRegressor
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.MambAttentionLSS
    :members:
    :inherited-members:

Transformer-Based Models
-------------------------

FTTransformer
~~~~~~~~~~~~~

.. autoclass:: deeptab.models.FTTransformerClassifier
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.FTTransformerRegressor
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.FTTransformerLSS
    :members:
    :inherited-members:

TabTransformer
~~~~~~~~~~~~~~

.. autoclass:: deeptab.models.TabTransformerClassifier
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.TabTransformerRegressor
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.TabTransformerLSS
    :members:
    :inherited-members:

SAINT
~~~~~

.. autoclass:: deeptab.models.SAINTClassifier
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.SAINTRegressor
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.SAINTLSS
    :members:
    :inherited-members:

MLP-Based Models
----------------

MLP
~~~

.. autoclass:: deeptab.models.MLPClassifier
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.MLPRegressor
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.MLPLSS
    :members:
    :inherited-members:

ResNet
~~~~~~

.. autoclass:: deeptab.models.ResNetClassifier
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.ResNetRegressor
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.ResNetLSS
    :members:
    :inherited-members:

TabM
~~~~

.. autoclass:: deeptab.models.TabMClassifier
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.TabMRegressor
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.TabMLSS
    :members:
    :inherited-members:

AutoInt
~~~~~~~

.. autoclass:: deeptab.models.AutoIntClassifier
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.AutoIntRegressor
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.AutoIntLSS
    :members:
    :inherited-members:

Tree-Based Models
-----------------

NODE
~~~~

.. autoclass:: deeptab.models.NODEClassifier
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.NODERegressor
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.NODELSS
    :members:
    :inherited-members:

ENODE
~~~~~

.. autoclass:: deeptab.models.ENODEClassifier
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.ENODERegressor
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.ENODELSS
    :members:
    :inherited-members:

NDTF
~~~~

.. autoclass:: deeptab.models.NDTFClassifier
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.NDTFRegressor
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.NDTFLSS
    :members:
    :inherited-members:

Specialized Models
------------------

TabR
~~~~

.. autoclass:: deeptab.models.TabRClassifier
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.TabRRegressor
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.TabRLSS
    :members:
    :inherited-members:

TabulaRNN
~~~~~~~~~

.. autoclass:: deeptab.models.TabulaRNNClassifier
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.TabulaRNNRegressor
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.TabulaRNNLSS
    :members:
    :inherited-members:

Experimental Models
-------------------

.. warning::

   The classes below live in ``deeptab.models.experimental``. Their API may
   change without a deprecation cycle. Import them explicitly::

      from deeptab.models.experimental import ModernNCAClassifier

   Always pin your DeepTab version when using experimental models.

ModernNCA
~~~~~~~~~

.. autoclass:: deeptab.models.experimental.ModernNCAClassifier
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.experimental.ModernNCARegressor
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.experimental.ModernNCALSS
    :members:
    :inherited-members:

Tangos
~~~~~~

.. autoclass:: deeptab.models.experimental.TangosClassifier
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.experimental.TangosRegressor
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.experimental.TangosLSS
    :members:
    :inherited-members:

Trompt
~~~~~~

.. autoclass:: deeptab.models.experimental.TromptClassifier
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.experimental.TromptRegressor
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.experimental.TromptLSS
    :members:
    :inherited-members:

Base Classes
------------

.. autoclass:: deeptab.models.SklearnBaseClassifier
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.SklearnBaseRegressor
    :members:
    :inherited-members:

.. autoclass:: deeptab.models.SklearnBaseLSS
    :members:
    :inherited-members:
