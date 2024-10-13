from dataclasses import dataclass, field

import torch


@dataclass
class QuantArgs:
    method: str = field(
        default="rtn",
        metadata={"choices": ["rtn", "gptq", "awq", "smoothquant"]},
    )
    implementation: str = field(
        default="native",
        metadata={"choices": ["native", "triton", "cuda"]},
    )
    precision: str = field(
        default=None,
        metadata={"choices": ["W8A8", "W8A16", "W16A16"]},
    )
    act_quant: bool = False
    training: bool = False
    device: torch.device = None

    # weight quantization arguments
    nbits: int = None
    group_size: int = None
    per_tensor: bool = False
    per_channel: bool = False
    quant_dtype: torch.dtype = None
    scale_dtype: torch.dtype = None
    zero_dtype: torch.dtype = None
    use_zero_point: bool = False

    # activation quantization arguments
    act_nbits: int = None
    act_group_size: int = None
    act_per_tensor: bool = False
    act_per_channel: bool = False
    act_quant_dtype: torch.dtype = None
    act_scale_dtype: torch.dtype = None
    act_zero_dtype: torch.dtype = None
    act_use_zero_point: bool = False

    # calibration arguments
    do_calibration: bool = None
    calib_dataset: str = None
    calib_batch_size: int = 1
    calib_nsamples: int = 128
    calib_maxlen: int = 512
    calib_text_column: str = "text"
    calib_data_shuffle: bool = False
    tokenizer_path: str = None

    # AWQ/Smoothquant arguments
    use_bidirectional_scale: bool = False
    smooth_alpha: float = 0.5

    def __post_init__(self):
        if self.nbits is None:
            raise ValueError("nbits must be specified")

        self.method = self.method.lower()

        if self.do_calibration is None and self.method in ["smoothquant", "awq"]:
            self.do_calibration = True
