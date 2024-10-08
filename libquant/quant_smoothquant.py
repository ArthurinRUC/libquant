import torch

from libquant.quant_rtn import quant_rtn


def quant_smoothquant(
    mat: torch.Tensor,
    nbits: int,
    group_size: int = None,
    per_tensor: bool = False,
    per_channel: bool = False,
    quant_dtype: torch.dtype = None,
    scale_dtype: torch.dtype = None,
    zero_dtype: torch.dtype = None,
    device: torch.device = None,
    use_zero_point: bool = False,
    is_linear_weight: bool = False,
    channel_wise_scale: torch.Tensor = None,
    use_bidirectional_scale: bool = False,
    post_layer_scale: torch.Tensor = None,
):
    if channel_wise_scale is None or not isinstance(channel_wise_scale, torch.Tensor):
        raise ValueError("channel_wise_scale should not be empty and should have a torch.Tensor type.")

    if channel_wise_scale.dim() == 1:
        channel_wise_scale = channel_wise_scale.diag()

    if channel_wise_scale.dim() != 2:
        raise ValueError("channel_wise_scale should have a 1D or 2D shape.")

    if use_bidirectional_scale:
        assert is_linear_weight is True
        assert post_layer_scale is not None

        if post_layer_scale is None or not isinstance(post_layer_scale, torch.Tensor):
            raise ValueError("post_layer_scale should not be empty and should have a torch.Tensor type.")

        if post_layer_scale.dim() == 1:
            post_layer_scale = post_layer_scale.diag()

        if post_layer_scale.dim() != 2:
            raise ValueError("post_layer_scale should have a 1D or 2D shape.")

        mat = post_layer_scale @ mat @ channel_wise_scale
    else:
        mat = mat @ channel_wise_scale

    return quant_rtn(
        mat,
        nbits,
        group_size,
        per_tensor,
        per_channel,
        quant_dtype,
        scale_dtype,
        zero_dtype,
        device,
        use_zero_point,
        is_linear_weight,
    )
