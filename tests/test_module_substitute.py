import torch
from torch import nn

from libquant.arguments import QuantArgs
from libquant.module import substitute_module


def test_module_substitute():
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, 16)

        def forward(self, x):
            return self.linear(x)

    model = MyModel()
    quant_args = QuantArgs(nbits=8, device="cuda")
    substitute_module(model, quant_args)

    assert model.linear.__class__.__name__ == "QuantLinear"
    assert model.linear.qweight.device == torch.device("cuda", index=0)

    data = torch.randn(16, 16, device="cuda")
    output = model(data)


if __name__ == "__main__":
    test_module_substitute()
