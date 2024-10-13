import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .arguments import QuantArgs
from .calibration import calibrate
from .module import QuantLinear, substitute_module


def quant_weight(model):
    # unwarp model
    while hasattr(model, "module"):
        model = model.module

    for name, module in model.named_children():
        if isinstance(module, QuantLinear):
            module.quant_weight()

        if len(list(module.children())) > 0:
            quant_weight(module)


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

    if quant_args.do_calibration:
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(quant_args.tokenizer_path, trust_remote_code=True)
        calibrate(
            model,
            data_path=quant_args.calib_dataset,
            tokenizer=tokenizer,
            batch_size=quant_args.calib_batch_size,
            nsamples=quant_args.calib_nsamples,
            max_length=quant_args.calib_maxlen,
            text_column=quant_args.calib_text_column,
            shuffle=quant_args.calib_data_shuffle,
            device=quant_args.device,
        )

    quant_weight(model)

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
