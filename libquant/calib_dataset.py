from datasets import load_dataset
from torch.utils.data import DataLoader


def load_calib_dataloader(data_path, batch_size, nsamples, maxlen, text_column="text"):
    dataset = load_dataset("jsonl", data_files=data_path, split="train")
    dataset = dataset.map(
        lambda x: {text_column: x[text_column][:maxlen]},
        batched=True,
        remove_columns=dataset.column_names,
    )
    dataset = dataset.select(range(nsamples))

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
