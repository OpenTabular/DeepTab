Configurations API
==================

.. currentmodule:: deeptab.configs

Base configs
------------

These three classes form the core of the split-config API and are shared across
**all** models.

.. autoclass:: deeptab.configs.TrainerConfig
   :members:
   :undoc-members:

.. autoclass:: deeptab.configs.PreprocessingConfig
   :members:
   :undoc-members:

.. autoclass:: deeptab.configs.BaseModelConfig
   :members:
   :undoc-members:

Model architecture configs
--------------------------

Each class below extends :class:`BaseModelConfig` and adds the hyperparameters
specific to one model family.

.. autoclass:: deeptab.configs.AutoIntConfig
   :members:
   :undoc-members:

.. autoclass:: deeptab.configs.ENODEConfig
   :members:
   :undoc-members:

.. autoclass:: deeptab.configs.FTTransformerConfig
   :members:
   :undoc-members:

.. autoclass:: deeptab.configs.MambaTabConfig
   :members:
   :undoc-members:

.. autoclass:: deeptab.configs.MambAttentionConfig
   :members:
   :undoc-members:

.. autoclass:: deeptab.configs.MambularConfig
   :members:
   :undoc-members:

.. autoclass:: deeptab.configs.MLPConfig
   :members:
   :undoc-members:

.. autoclass:: deeptab.configs.NDTFConfig
   :members:
   :undoc-members:

.. autoclass:: deeptab.configs.NODEConfig
   :members:
   :undoc-members:

.. autoclass:: deeptab.configs.ResNetConfig
   :members:
   :undoc-members:

.. autoclass:: deeptab.configs.SAINTConfig
   :members:
   :undoc-members:

.. autoclass:: deeptab.configs.TabMConfig
   :members:
   :undoc-members:

.. autoclass:: deeptab.configs.TabRConfig
   :members:
   :undoc-members:

.. autoclass:: deeptab.configs.TabTransformerConfig
   :members:
   :undoc-members:

.. autoclass:: deeptab.configs.TabulaRNNConfig
   :members:
   :undoc-members:

Experimental model configs
--------------------------

.. autoclass:: deeptab.configs.ModernNCAConfig
   :members:
   :undoc-members:

.. autoclass:: deeptab.configs.TangosConfig
   :members:
   :undoc-members:

.. autoclass:: deeptab.configs.TromptConfig
   :members:
   :undoc-members:
