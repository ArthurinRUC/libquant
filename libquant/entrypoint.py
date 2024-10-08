import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .arguments import QuantArgs
from .module import substitute_module


def quant(
    model: torch.nn.Module,
    quant_args: QuantArgs = None,
    tokenizer: AutoTokenizer = None,
    method: str = "rtn",
    implementation: str = "native",
    training: bool = False,
    **kwargs,
):
    if quant_args is None:
        quant_args = QuantArgs(method=method, implementation=implementation, training=training, **kwargs)
    substitute_module(model, quant_args)
    return model


class QuantModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, tokenizer: AutoTokenizer, quant_args: QuantArgs):
        super().__init__()
        self.model = quant(model, quant_args, tokenizer)
        self.tokenizer = tokenizer
        self.quant_args = quant_args

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, model_path: str, quant_args: QuantArgs, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return cls(model, tokenizer, quant_args)
