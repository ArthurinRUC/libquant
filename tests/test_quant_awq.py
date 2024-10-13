import torch
from transformers import AutoTokenizer

from libquant.calibration import calibrate


class ModelForTest(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(200000, 3)
        self.linear1 = torch.nn.Linear(3, 4)
        self.layernorm = torch.nn.LayerNorm(4)
        self.linear2 = torch.nn.Linear(4, 5)

    def forward(self, input_ids, **kwargs):
        return self.linear2(self.layernorm(self.linear1(self.embedding(input_ids))))

    @property
    def device(self):
        return self.linear1.weight.device


@torch.inference_mode()
def test_calib_awq(data_path: str, tokenizer_path: str, batch_size: int, max_length: int, nsamples: int, device=None):
    model = ModelForTest()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    scales = calibrate(
        model, data_path, tokenizer, batch_size, nsamples, max_length, device=device, debug_to_get_scale=True
    )
    print(scales)


if __name__ == "__main__":
    test_calib_awq(
        data_path="/data/HuggingFace/datasets/Alpaca_Pretrain/eval/test.jsonl",
        tokenizer_path="/data/HuggingFace/models/Qwen2-7B-Instruct",
        batch_size=4,
        max_length=256,
        nsamples=5000,
    )
