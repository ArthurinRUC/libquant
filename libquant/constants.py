import torch

from .module import QuantLinear


LINEAR_MODULES = (torch.nn.Linear, QuantLinear)
REPLACE_LINEAR_MODULES = (torch.nn.Linear, QuantLinear)
