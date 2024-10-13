import torch
import torch.nn.functional as F
from torch import nn

from .arguments import QuantArgs
from .constants import REPLACE_LINEAR_MODULES
from .quant_awq import quant_awq
from .quant_rtn import dequant_rtn, quant_rtn
from .quant_smoothquant import quant_smoothquant


QUANT_FUNCTIONS = {"rtn": quant_rtn, "awq": quant_awq, "smoothquant": quant_smoothquant}
DEQUANT_FUNCTIONS = {"rtn": dequant_rtn, "awq": dequant_rtn, "smoothquant": dequant_rtn}


def substitute_module(model, quant_args):
    # unwarp model
    while hasattr(model, "module"):
        model = model.module

    for name, module in model.named_children():
        if isinstance(module, REPLACE_LINEAR_MODULES):
            model._modules[name] = substitute_linear(module, quant_args)

        if len(list(module.children())) > 0:
            substitute_module(module, quant_args)


def substitute_linear(module, quant_args):
    return QuantLinear.from_linear(module, quant_args)


class QuantLinear(nn.Module):
    def __init__(
        self,
        quant_args: QuantArgs,
        weight: torch.Tensor = None,
        qweight: torch.Tensor = None,
        scale: torch.Tensor = None,
        zero_point: torch.Tensor = None,
        bias: torch.Tensor = None,
        **kwargs,
    ):
        super().__init__()
        self.args = quant_args
        self.register_buffer("weight", weight)
        self.register_buffer("qweight", qweight)
        self.register_buffer("bias", bias)
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)

        self.quant_fn = QUANT_FUNCTIONS[self.args.method]
        self.dequant_fn = DEQUANT_FUNCTIONS[self.args.method]

        if self.args.do_calibration:
            self.register_buffer("act_scale", None)
            self.register_buffer("weight_scale", None)
            self.register_buffer("smooth_scale", None)

    @classmethod
    def from_linear(cls: "QuantLinear", module: nn.Linear, quant_args: QuantArgs, **kwargs):
        device = quant_args.device
        weight, bias = module.weight, module.bias
        if quant_args.device:
            weight = weight.to(device)
        if bias is not None and quant_args.device:
            bias = bias.to(device)

        return cls(quant_args, weight=weight, bias=bias, **kwargs)

    def quant_weight(self, **kwargs):
        output = self.quant_fn(
            self.weight,
            nbits=self.args.nbits,
            group_size=self.args.group_size,
            per_tensor=self.args.per_tensor,
            per_channel=self.args.per_channel,
            quant_dtype=self.args.quant_dtype,
            scale_dtype=self.args.scale_dtype,
            zero_dtype=self.args.zero_dtype,
            device=self.args.device,
            use_zero_point=self.args.use_zero_point,
            is_linear_weight=True,
            **kwargs,
        )

        self.qweight = output["quant_tensor"]
        self.scale = output["scale"]
        self.zero_point = output.get("zero_point", None)

        self.weight = None

    def forward(self, x):
        if self.args.training:
            return NotImplemented

        with torch.inference_mode():
            if self.args.act_quant:
                # quantize activation online
                x_output = self.quant_fn(
                    x,
                    nbits=self.args.act_nbits,
                    group_size=self.args.act_group_size,
                    per_tensor=self.args.act_per_tensor,
                    per_channel=self.args.act_per_channel,
                    quant_dtype=self.args.act_quant_dtype,
                    scale_dtype=self.args.act_scale_dtype,
                    zero_dtype=self.args.act_zero_dtype,
                    device=self.args.device,
                    use_zero_point=self.args.act_use_zero_point,
                )

                # Ideally, we should dequantize the activation and weight AFTER matrix multiplication.
                # However, mathematically, it is challenging to dequantize both the weight and activation AFTER gemm operation.
                # Different types of quantization for the weight and activation (e.g., using zero points or per-token/per-channel quantization)
                # add more complexity to the code. Therefore, we have to perform *fake quantization* here.
                x_quant = self.dequant_fn(
                    x_output["quant_tensor"],
                    scale=x_output["scale"],
                    zero_point=x_output.get("zero_point", None),
                    group_size=self.args.act_group_size,
                    per_tensor=self.args.act_per_tensor,
                    per_channel=self.args.act_per_channel,
                    is_linear_weight=False,
                )

                w_quant = self.dequant_fn(
                    self.qweight,
                    scale=self.scale,
                    zero_point=self.zero_point,
                    group_size=self.args.group_size,
                    per_tensor=self.args.per_tensor,
                    per_channel=self.args.per_channel,
                    is_linear_weight=True,
                )

                # Here we upcast X and W to float32 (i.e. fake quantization)
                # since torch does not support int8 x int8 -> int32 multiplication on CUDA yet
                x_quant = x_quant.to(torch.float32)
                w_quant = w_quant.to(torch.float32)

                # forward pass
                return F.linear(x_quant, w_quant, self.bias)

            else:
                # weight dequantization
                weight = self.dequant_fn(
                    self.qweight,
                    scale=self.scale,
                    zero_point=self.zero_point,
                    group_size=self.args.group_size,
                    per_tensor=self.args.per_tensor,
                    per_channel=self.args.per_channel,
                    is_linear_weight=True,
                )
                return F.linear(x, weight, self.bias)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.qweight.shape[1]}, {self.qweight.shape[0]})"
