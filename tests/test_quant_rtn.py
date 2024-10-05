import pytest
import torch

from libquant.quant_rtn import dequant_rtn, quant_rtn


@pytest.mark.parametrize("use_zero_point", [True, False])
@pytest.mark.parametrize("per_channel", [True, False])
@pytest.mark.parametrize("per_tensor", [True, False])
@pytest.mark.parametrize("group_size", [None, 4, 16])
@pytest.mark.parametrize(
    "data_shape", [(1, 16, 16), (1, 128, 128), (1, 256, 1024), (16, 256, 1024), (16, 16, 128, 128)]
)
@pytest.mark.parametrize("nbits", [6, 8, 16])
def test_quant_rtn(
    nbits: int, data_shape: tuple, group_size: int, per_tensor: bool, per_channel: bool, use_zero_point: bool
):
    device = torch.device("cuda")
    mat = torch.randn(*data_shape, dtype=torch.float32, device=device)
    output = quant_rtn(
        mat=mat,
        nbits=nbits,
        group_size=group_size,
        per_tensor=per_tensor,
        per_channel=per_channel,
        device=device,
        use_zero_point=use_zero_point,
    )
    dq_mat = dequant_rtn(
        output["quant_tensor"],
        output["scale"],
        output["zero_point"] if use_zero_point else None,
        group_size=group_size,
        per_tensor=per_tensor,
        per_channel=per_channel,
    )

    assert torch.allclose(mat, dq_mat, atol=1e-1)


if __name__ == "__main__":
    test_quant_rtn()
