import triton
import triton.language as tl


@triton.jit
def quant_rtn_triton(
    mat: tl.tensor,
    scale: tl.tensor,
    zero_point: tl.tensor,
    quant_dim: int,
    nbits: int,
    per_channel: bool,
    per_tensor: bool,
    use_zero_point: bool,
    group_size: int,
    scale_dtype: tl.dtype,
    zero_dtype: tl.dtype,
    quant_dtype: tl.dtype,
    device: tl.dtype,
) -> tl.Tuple[tl.tensor, tl.tensor]:
    origin_shape = mat.shape
    if group_size is None:
        group_size = origin_shape[-1]
    mat = mat.reshape(-1, group_size)
    if use_zero_point:
        qmin, qmax = 0, 2**nbits - 1
        xmin, xmax = tl.min(mat, axis=quant_dim), tl.max(mat, axis=quant_dim)
        scale = (xmax - xmin).div(qmax).clamp(min=1e-5).to(scale_dtype).to(device)
        zero_point = (-xmin / scale).to(zero_dtype).to(device)
        if not per_channel and not per_tensor:
            scale = scale.unsqueeze(1)
            zero_point = zero_point.unsqueeze(1)
        x = mat.div(scale).add(zero_point).round().clamp(qmin, qmax).to(quant_dtype).to(device)
    else:
        qmin, qmax = -(2 ** (nbits - 1)), 2 ** (nbits - 1) - 1
        xabsmax = tl.max(tl.abs(mat), axis=quant_dim)
        scale = xabsmax.div(qmax).clamp(min=1e-5).to(scale_dtype).to(device)
        if not per_channel and not per_tensor:
            scale = scale.unsqueeze(1)
        x = mat.div(scale).round().clamp(qmin, qmax).to(quant_dtype).to(device)

    return x.reshape(origin_shape), scale
