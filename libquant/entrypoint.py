import torch

from .arguments import QuantArgs
from .module import substitute_module


def quant(
    model: torch.nn.Module,
    quant_args: QuantArgs = None,
    method: str = "rtn",
    implementation: str = "native",
    training: bool = False,
    **kwargs,
):
    if quant_args is None:
        quant_args = QuantArgs(method=method, implementation=implementation, training=training, **kwargs)
    substitute_module(model, quant_args)
    return model
