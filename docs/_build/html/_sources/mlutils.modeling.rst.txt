mlutils.modeling
================

Public entry point for PyTorch modeling symbols in mlutils.

When PyTorch is available, this module re-exports ``NNmodel``, ``AutoEncoderModel``, ``torchModel``, and ``instantiate_models`` from ``mlutils.modeling_torch``.
Otherwise those names are ``None`` and ``instantiate_models`` raises ``ImportError``.

.. automodule:: msdlib.mlutils.modeling
   :members:
   :undoc-members:
   :show-inheritance:
