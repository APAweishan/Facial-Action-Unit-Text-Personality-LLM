import os
import json
import torch
import random
import numpy as np
import argparse
from typing import Dict, Any, List

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
from tensorboardX import SummaryWriter

from dataset_module import GeneralDataset, formatting_func
from model_module import LLMWithRegressionHead

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# -----------------------------
# Shared helpers
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_json_samples(paths):
    if isinstance(paths, str):
        paths = [paths]
    samples = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        for sample_id, sample_content in raw_data.items():
            sample_content["id"] = sample_id
            samples.append(sample_content)
    return samples


def convert_output_to_float(samples):
    for sample in samples:
        try:
            sample["label_float"] = float(sample["output"])
        except Exception as e:
            raise ValueError(f"Failed to convert output to float: {sample['output']}") from e


def val(step, model, valloader, device, writer, q):
    model.eval()
    val_total_loss = 0
    with torch.no_grad():
        loop = tqdm(valloader, desc=f"[Validation q={q}]")
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label_float"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = torch.nn.functional.mse_loss(outputs, labels)
            val_total_loss += loss.item()
    avg_loss = val_total_loss / len(valloader)
    writer.add_scalar(f"Loss/val_{q}", avg_loss, step)
    print(f"[Validation] Step {step} | q={q} | Loss: {avg_loss:.4f}")
    model.train()
    return avg_loss


# -----------------------------
# Training pipeline
# -----------------------------
def run_training(
    q: str,
    save_dir: str,
    output_dir: str,
    model_path: str,
    model_name: str = "Llama3",
    suffix: str = "score_prediction",
    max_steps: int = 3000,
    batch_size: int = 2,
    lr: float = 3e-4,
    warmup_steps: int = 100,
    patience: int = 5,
    save_step: int = 300,
    context_length: int = 4096,
    accumulation_steps: int = 4,
    seed: int = 42,
):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    base_model = get_peft_model(base_model, lora_config)

    model = LLMWithRegressionHead(base_model, tokenizer).to(device)

    train_samples = load_json_samples([os.path.join(output_dir, f"{q}_train_{suffix}.json")])
    val_samples = load_json_samples([os.path.join(output_dir, f"{q}_valid_{suffix}.json")])
    convert_output_to_float(train_samples)
    convert_output_to_float(val_samples)

    class ExtendedDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, samples):
            self.base_dataset = base_dataset
            self.samples = samples

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, idx):
            base_item = self.base_dataset[idx]
            base_item["label_float"] = torch.tensor(self.samples[idx]["label_float"], dtype=torch.float)
            return base_item

    base_trainset = GeneralDataset(train_samples, tokenizer, formatting_func, context_length, add_special_tokens=True)
    base_valset = GeneralDataset(val_samples, tokenizer, formatting_func, context_length, add_special_tokens=True)

    def collate_fn(batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        ids = [item["id"] for item in batch]
        labels = [item["label_float"] for item in batch]

        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "label_float": torch.stack(labels),
            "id": ids,
        }

    trainset = ExtendedDataset(base_trainset, train_samples)
    valset = ExtendedDataset(base_valset, val_samples)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps)
    writer = SummaryWriter()
    scaler = torch.amp.GradScaler("cuda")

    best_val_loss = float("inf")
    early_stop_counter = 0
    step = 0
    model.train()

    for epoch in range(100):
        for batch_idx, batch in enumerate(tqdm(trainloader, desc=f"[Train Epoch {epoch} q={q}]")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label_float"].to(device)

            with torch.amp.autocast("cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = torch.nn.functional.mse_loss(outputs, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                writer.add_scalar(f"Loss/train_{q}", loss.item() * accumulation_steps, step)
                step += 1

                if step % save_step == 0:
                    val_loss = val(step, model, valloader, device, writer, q)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        early_stop_counter = 0
                        print(f"✅ Saving best model at step {step}, val loss: {val_loss:.4f}")
                        model.backbone.save_pretrained(f"{save_dir}/best_{q}")
                        tokenizer.save_pretrained(f"{save_dir}/best_{q}")
                        torch.save(model.regression_head.state_dict(), f"{save_dir}/best_{q}/regression_head.pt")
                    else:
                        early_stop_counter += 1
                        if early_stop_counter >= patience:
                            print("⏹️ Early stopping triggered.")
                            writer.close()
                            return
                if step >= max_steps:
                    print("⏹️ Max steps reached.")
                    writer.close()
                    return

        val_loss = val(step, model, valloader, device, writer, q)
        print(f"📉 [Epoch {epoch}] validation loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            print(f"✅ Saving best model at epoch {epoch}, val loss: {val_loss:.4f}")
            model.backbone.save_pretrained(f"{save_dir}/best_{q}")
            tokenizer.save_pretrained(f"{save_dir}/best_{q}")
            torch.save(model.regression_head.state_dict(), f"{save_dir}/best_{q}/regression_head.pt")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("⏹️ Early stopping triggered.")
                writer.close()
                return


# -----------------------------
# CLI
# -----------------------------
MODES = {
    "full": {
        "save_dir": "./model_save/regression_lora_output_selected_AU_last_{model}",
        "output_dir": "/root/autodl-tmp/Prolific/data_json",
    },
    # Ablations
    "all_au": {
        "save_dir": "./model_save/ablation_all_AU_{model}",  # w/o AU selection
        "output_dir": "/root/autodl-tmp/Prolific/ablation_data/ablation_all_AU",
    },
    "only_text": {
        "save_dir": "./model_save/ablation_only_text_{model}",  # w/o AU description
        "output_dir": "/root/autodl-tmp/Prolific/ablation_data/ablation_only_text",
    },
    "only_au": {
        "save_dir": "./model_save/ablation_only_AU_{model}",  # w/o text
        "output_dir": "/root/autodl-tmp/Prolific/ablation_data/ablation_only_AU",
    },
}

MODE_ALIASES = {
    "ablation_all_AU": "all_au",
    "ablation_only_text": "only_text",
    "ablation_only_AU": "only_au",
}

ALL_MODE_CHOICES = sorted(set(MODES.keys()) | set(MODE_ALIASES.keys()))


def parse_args():
    parser = argparse.ArgumentParser(description="Unified training & ablation runner.")
    parser.add_argument("--mode", choices=ALL_MODE_CHOICES, default="full", help="Choose experiment variant.")
    parser.add_argument("--q", type=str, default="q3", help="Question id (e.g., q3).")
    parser.add_argument("--model-path", type=str, default="/root/autodl-tmp/LLM-Research/Meta-Llama-3___1-8B-Instruct")
    parser.add_argument("--model-name", type=str, default="Llama3")
    parser.add_argument("--suffix", type=str, default="score_prediction")
    parser.add_argument("--save-dir", type=str, default=None, help="Override save dir.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override data dir.")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--save-step", type=int, default=300)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    mode = MODE_ALIASES.get(args.mode, args.mode)
    mode_cfg = MODES[mode]
    save_dir = args.save_dir or mode_cfg["save_dir"].format(model=args.model_name)
    output_dir = args.output_dir or mode_cfg["output_dir"]

    run_training(
        q=args.q,
        save_dir=save_dir,
        output_dir=output_dir,
        model_path=args.model_path,
        model_name=args.model_name,
        suffix=args.suffix,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        patience=args.patience,
        save_step=args.save_step,
        context_length=args.context_length,
        accumulation_steps=args.accumulation_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
