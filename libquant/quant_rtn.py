import torch


@torch.no_grad()
def quant_rtn(
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
):
    origin_shape = mat.shape
    mat = mat.reshape(-1, origin_shape[-1])
    quant_dim = 1 ^ per_channel ^ is_linear_weight

    quant_dim_length = mat.shape[quant_dim]
    if per_tensor:
        group_size = mat.nelement()
    elif group_size is None:
        group_size = quant_dim_length
    elif quant_dim_length % group_size != 0:
        raise ValueError(
            f"Quantization dimension length should be divisible by group_size, got {quant_dim_length} and {group_size}"
        )

    if scale_dtype is None:
        scale_dtype = mat.dtype

    if zero_dtype is None:
        zero_dtype = torch.float32

    if quant_dtype is None:
        if nbits < 8:
            quant_dtype = torch.int8
        elif nbits == 8:
            quant_dtype = torch.uint8 if use_zero_point else torch.int8
        elif nbits < 16:
            quant_dtype = torch.int16
        elif nbits == 16:
            quant_dtype = torch.uint16 if use_zero_point else torch.int16
        else:
            quant_dtype = torch.int32

    if device is None:
        device = mat.device
    else:
        mat = mat.to(device)

    # TODO: Try to remove the transpose operation to make quantization faster
    if quant_dim == 0 and not per_tensor:
        mat.t_()

    mat = mat.reshape(-1, group_size)

    if use_zero_point:
        # asymmetric quantization
        qmin, qmax = 0, 2**nbits - 1
        xmin, xmax = mat.amin(1), mat.amax(1)
        scale = (xmax - xmin).clamp(min=1e-5).div(qmax).to(scale_dtype)
        zero_point = (-xmin / scale).to(zero_dtype)
        if not per_tensor:
            scale.unsqueeze_(1)
            zero_point.unsqueeze_(1)
        x = mat.div(scale).add(zero_point).round().clamp(qmin, qmax).to(quant_dtype)
    else:
        # symmetric quantization
        qmin, qmax = -(2 ** (nbits - 1)), 2 ** (nbits - 1) - 1
        xabsmax = mat.abs().amax(1)
        scale = xabsmax.clamp(min=1e-5).div(qmax).to(scale_dtype)
        if not per_tensor:
            scale.unsqueeze_(1)
        x = mat.div(scale).round().clamp(qmin, qmax).to(quant_dtype)

    x = x.reshape(-1, quant_dim_length)

    # TODO: Try to remove the transpose operation to make quantization faster
    if quant_dim == 0 and not per_tensor:
        x.t_()

    output_dict = {
        "quant_tensor": x.reshape(origin_shape),
        "scale": scale,
    }

    if use_zero_point:
        output_dict["zero_point"] = zero_point

    return output_dict


@torch.no_grad()
def dequant_rtn(
    mat: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor = None,
    group_size: int = None,
    per_tensor: bool = False,
    per_channel: bool = False,
    is_linear_weight: bool = False,
):
    origin_shape = mat.shape
    mat = mat.reshape(-1, origin_shape[-1])
    quant_dim = 1 ^ per_channel ^ is_linear_weight

    quant_dim_length = mat.shape[quant_dim]
    if per_tensor:
        group_size = mat.nelement()
    elif group_size is None:
        group_size = quant_dim_length
    elif quant_dim_length % group_size != 0:
        raise ValueError(
            f"Quantization dimension length should be divisible by group_size, got {quant_dim_length} and {group_size}"
        )

    # TODO: Try to remove the transpose operation to make dequantization faster
    if quant_dim == 0 and not per_tensor:
        mat.t_()

    mat = mat.reshape(-1, group_size)

    if zero_point is None:
        mat = mat * scale
    else:
        mat = (mat - zero_point) * scale

    mat = mat.reshape(-1, quant_dim_length)

    # TODO: Try to remove the transpose operation to make dequantization faster
    if quant_dim == 0 and not per_tensor:
        mat.t_()

    return mat.reshape(origin_shape)
