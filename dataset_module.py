import torch
from torch.utils.data import Dataset
from typing import List, Dict, Callable, Iterable, Any
from tqdm import tqdm

def formatting_func(sample: dict, tokenizer, include_output=False):
    if "input" in sample and sample["input"] != "":
        prompt = f"""

{sample["input"]}. {sample["instruction"]}"""
    else:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

{sample["instruction"]}"""

    messages = [
        {"role": "user", "content": prompt}
    ]

    if include_output:
        messages.append({"role": "assistant", "content": sample["output"]})

    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return text


def tokenize_dataset(samples: Iterable[Any],
                     tokenizer,
                     formating_func: Callable,
                     max_length: int = -1,
                     add_special_tokens: bool = False):

    texts = [formating_func(sample, tokenizer) for sample in samples]
    loop = tqdm(enumerate(texts), total=len(texts))
    loop.set_description("Tokenizing")
    encodings = []
    for _, text in loop:
        if max_length > 0:
            encoding = tokenizer.encode(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                add_special_tokens=add_special_tokens
            ).squeeze()
        else:
            encoding = tokenizer.encode(
                text,
                return_tensors="pt",
                add_special_tokens=add_special_tokens
            ).squeeze()
        encodings.append(encoding)

    return encodings


def padding_handler(encodings: List[torch.Tensor]):
    return encodings


def packed_handler(encodings: List[torch.Tensor], max_length: int):
    ## concatenate all tensors
    encoding = torch.cat(encodings, dim=0)

    ## split into chunks of max_length
    encodings = []
    for i in range(0, encoding.size(0), max_length):
        encodings.append(encoding[i:i+max_length])

    return encodings


def concat_handler(encodings: List[torch.Tensor], max_length: int):
    ## sort by length descending
    encodings = sorted(encodings, key=lambda x: x.shape[0], reverse=True)
    encodings_concated = []
    i = 0
    j = len(encodings) - 1
    current = encodings[i]
    while i < j:
        right_one = encodings[j]
        if current.shape[0] + right_one.shape[0] <= max_length:
            current = torch.cat([current, right_one], dim=0)
            j -= 1
        else:
            encodings_concated.append(current)
            i += 1
            current = encodings[i]
    encodings_concated.append(current)
    encodings = encodings_concated

    return encodings


class GeneralDataset(Dataset):

    def __init__(self,
                 samples: List[Dict[str, str]],
                 tokenizer,
                 formating_func: Callable,
                 max_length: int,
                 mode: str = "padding",
                 add_special_tokens: bool = False):

        if mode == "packed":
            encodings = tokenize_dataset(samples, tokenizer, formating_func, -1, add_special_tokens)
            encodings = packed_handler(encodings, max_length)
        else:
            encodings = tokenize_dataset(samples, tokenizer, formating_func, max_length, add_special_tokens)
            if mode == "concat":
                encodings = concat_handler(encodings, max_length)
            elif mode == "padding":
                encodings = padding_handler(encodings)
            else:
                raise ValueError(f"Invalid mode: {mode}")

        inputs = []
        for encoding in encodings:
            inputs.append({
                "input_ids": encoding,
                "attention_mask": torch.ones_like(encoding)
            })

        self.inputs = inputs
        self.ids = [sample.get("id", str(i)) for i, sample in enumerate(samples)]  # 更安全

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        item = self.inputs[idx]
        item["id"] = self.ids[idx]
        return item
