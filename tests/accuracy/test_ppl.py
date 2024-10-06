import os
from functools import partial

import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding


def tokenize(examples, tokenizer, max_length=1024):
    text = examples["text"]
    result = tokenizer(text, max_length=max_length, truncation=True)
    return result


class PretrainDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=1024):
        data_file_list = sorted(
            [
                os.path.join(data_dir, path)
                for path in os.listdir(data_dir)
                if path.endswith(".json") or path.endswith(".jsonl")
            ]
        )
        dataset_list = []
        for data_path in data_file_list:
            dataset = load_dataset("json", data_files=data_path, split="train")
            partial_dataset = dataset.map(
                partial(tokenize, tokenizer=tokenizer, max_length=max_length),
                batched=True,
                batch_size=100,
                remove_columns=dataset.column_names,
                num_proc=10,
            )
            dataset_list.append(partial_dataset)
        self.dataset = concatenate_datasets(dataset_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


@torch.inference_mode()
def test_ppl_alpaca():
    model_path = "/home/arthur/Meta-Llama-3-8B/"
    data_dir = "/mnt/d/linuxdata/HuggingFace/datasets/val"
    device = "cuda"
    batch_size = 1
    iters = 5000

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()

    # quant(model, nbits=8, per_tensor=True)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = PretrainDataset(data_dir, tokenizer, 1024)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )

    dataloader = iter(dataloader)

    losses = []
    tokens = []

    for idx in tqdm(range(iters)):
        inputs = next(dataloader)
        # get batch data and move to device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        bs, seqlen = input_ids.shape

        # model forward path
        outputs = model(input_ids, attention_mask, use_cache=False)
        logits = outputs.logits

        # compute loss and ppl
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss_fn = torch.nn.functional.cross_entropy
        loss = loss_fn(
            input=shift_logits.view(-1, shift_logits.size(-1)),
            target=shift_labels.view(-1),
            ignore_index=tokenizer.pad_token_id,
            reduction="none",
        )
        loss = loss.view(bs, seqlen - 1)

        sample_total_loss = loss.sum(dim=1)
        sample_output_tokens = attention_mask.sum(dim=1) - 1

        losses.append(sample_total_loss.to("cpu"))
        tokens.append(sample_output_tokens.to("cpu"))

    losses = torch.cat(losses)
    tokens = torch.cat(tokens)
    print(f"Dataset Perplexity: {torch.exp(losses.sum() / tokens.sum())}")
