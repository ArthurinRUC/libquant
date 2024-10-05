# libquant

libquant is a library for low-bit machine learning model quantization, making them more efficient and faster without significant loss in accuracy. It supports various quantization methods, types, and granularities, and provides quantization implementations like native Pytorch, Triton, and CUDA.

## Features

- **Quantization Methods**: RTN, GPTQ, AWQ quantization
- **Quantization Types**: weight, activation quantization
- **Granularity**: per-tensor, per-token, per-channel, per-group quantization
- **Implementations**: PyTorch native, Triton, CUDA

## Installation

To install libquant from source, run:

```bash
git clone https://github.com/ArthurinRUC/libquant.git
cd libquant
pip install -e .
```

## Usage

To start using libquant, just add 2 lines of code to quantize your model.

```diff
from transformers import AutoModelForCausalLM
+ from libquant import quant

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
+ quant(model, nbits=8)
```

You can use `QuantArgs` to quantize your model with a custom quantization configuration.

```diff
from transformers import AutoModelForCausalLM
+ from libquant import quant, QuantArgs

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

+ quant_args = QuantArgs(method="rtn", nbits=8, group_size=128, per_channel=True)

+ quant(model, quant_args)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
