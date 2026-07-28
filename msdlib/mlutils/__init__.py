"""
Machine learning utilities for training and evaluating multiple model types.

Sklearn-compatible helpers such as ``train_with_data``, ``evaluate_with_data``, ``store_models``, and ``load_models``
work without PyTorch. Deep learning symbols require PyTorch; use ``TORCH_AVAILABLE`` to check availability.

Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""


from .utils import (
    TORCH_AVAILABLE,
    DataSet,
    get_factors,
    define_layers,
    store_models,
    load_models,
    train_with_data,
    evaluate_with_data,
)
from .modeling import (
    NNmodel,
    AutoEncoderModel,
    torchModel,
    instantiate_models,
)

__all__ = [
    "TORCH_AVAILABLE",
    "NNmodel",
    "AutoEncoderModel",
    "torchModel",
    "DataSet",
    "get_factors",
    "define_layers",
    "store_models",
    "load_models",
    "train_with_data",
    "evaluate_with_data",
    "instantiate_models",
]
