"""
Public entry point for PyTorch modeling symbols in mlutils.

Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""

from .utils import TORCH_AVAILABLE

NNmodel = None
AutoEncoderModel = None
torchModel = None


def instantiate_models():
    """
    Example factory for models passed to ``train_with_data``.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    raise ImportError(
        "instantiate_models requires PyTorch. Install torch to use this function."
    )


if TORCH_AVAILABLE:
    from .modeling_torch import (
        NNmodel,
        AutoEncoderModel,
        torchModel,
        instantiate_models,
    )
