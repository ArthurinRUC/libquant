from collections import defaultdict
from functools import partial

import datasets
import torch
from accelerate import PartialState
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

from .constants import LINEAR_MODULES


FILEEXT2TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text",
}


state = PartialState()


def tokenize_data(examples: dict, tokenizer: PreTrainedTokenizer, max_length: int, text_column: str):
    inputs = defaultdict(list)

    for i in range(len(examples[text_column])):
        input_ids = tokenizer.encode(examples[text_column][i])[:max_length]
        inputs["input_ids"].append(input_ids)
        inputs["attention_mask"].append([1] * len(input_ids))

    return inputs


class CalibDataset(Dataset):
    def __init__(self, dataset: datasets.Dataset, tokenizer: PreTrainedTokenizer, max_length: int, text_column: str):
        super().__init__()

        preprocess_func = partial(tokenize_data, tokenizer=tokenizer, max_length=max_length, text_column=text_column)
        origin_column_names = list(next(iter(dataset)).keys())

        with state.local_main_process_first():
            self.dataset = dataset.map(preprocess_func, batched=True, remove_columns=origin_column_names)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> dict[str, list[int]]:
        return self.dataset[i]


def get_calib_dataloader(
    data_path: str | list[str],
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    nsamples: int,
    max_length: int,
    text_column: str = "text",
    shuffle: bool = False,
):
    data_type = FILEEXT2TYPE.get(data_path.split(".")[-1], None)
    dataset = load_dataset(data_type, data_files=data_path, split=f"train[:{nsamples}]")
    dataset = CalibDataset(dataset, tokenizer, max_length, text_column)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, collate_fn=collator)

    return dataloader


def calibrate(
    model: torch.nn.Module,
    data_path: str | list[str],
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    nsamples: int,
    max_length: int,
    text_column: str = "text",
    shuffle: bool = False,
    device: torch.device = None,
    debug_to_get_scale: bool = False,
):
    # TODO: reduce-scatter DP act_scales for multi-device calibration

    act_forward_hooks = []
    weight_forward_hooks = []

    act_scales = defaultdict() if debug_to_get_scale else None
    weight_scales = defaultdict() if debug_to_get_scale else None

    def _calculate_act_scales(
        module: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, name: str = None, scales: dict = None
    ):
        if isinstance(x, tuple):
            x = x[0]

        # Calculate per-channel activation scales
        s = x.view(-1, x.shape[-1]).amax(dim=0).float()

        # Only for debug, in reality we prefer to store scales in quantlinear module.
        if scales is not None:
            if name in scales:
                scales[name] = torch.max(s, scales[name])
            else:
                scales[name] = s
        else:
            if module.act_scale is not None:
                module.act_scale = torch.max(s, module.act_scale)
            else:
                module.act_scale = s

    def _calculate_weight_scales(
        module: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, name: str = None, scales: dict = None
    ):
        w: torch.Tensor = module.weight

        # Calculate channel-wise weight scales
        s = w.amax(dim=0).float()

        # Only for debug, in reality we prefer to store scales in quantlinear module.
        if scales is not None:
            scales[name] = s
        else:
            module.weight_scale = s

    def register_calib_hook():
        for name, module in model.named_modules():
            if isinstance(module, LINEAR_MODULES):
                act_forward_hooks.append(
                    module.register_forward_hook(partial(_calculate_act_scales, name=name, scales=act_scales))
                )
                weight_forward_hooks.append(
                    module.register_forward_hook(partial(_calculate_weight_scales, name=name, scales=weight_scales))
                )

    def remove_calib_act_hook():
        for h in act_forward_hooks:
            h.remove()

    def remove_calib_weight_hook():
        for h in weight_forward_hooks:
            h.remove()

    register_calib_hook()

    dataloader = get_calib_dataloader(data_path, tokenizer, batch_size, nsamples, max_length, text_column, shuffle)

    if device:
        model = model.to(device)
    else:
        device = model.device

    get_weight_scale = False
    for data in tqdm(dataloader, desc="Data Calibration"):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        _ = model(input_ids=input_ids, attention_mask=attention_mask)

        if not get_weight_scale:
            get_weight_scale = True
            remove_calib_weight_hook()

    remove_calib_act_hook()

    return act_scales, weight_scales
