import os
import json
import torch
import numpy as np
import argparse
from typing import Dict

from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import mean_squared_error, mean_absolute_error

from dataset_module import GeneralDataset, formatting_func
from model_module import LLMWithRegressionHead


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json_samples(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    samples = []
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


class ExtendedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, samples):
        self.base_dataset = base_dataset
        self.samples = samples

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        item["label_float"] = torch.tensor(self.samples[idx]["label_float"], dtype=torch.float)
        return item


def evaluate(model, dataloader, device, q, results_path: str):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    results = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label_float"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = torch.nn.functional.mse_loss(outputs, labels)
            total_loss += loss.item()

            preds_cpu = outputs.cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            batch_ids = batch["id"]

            all_preds.extend(preds_cpu)
            all_labels.extend(labels_cpu)

            for j in range(len(batch_ids)):
                results[batch_ids[j]] = {
                    "pred": f"{float(preds_cpu[j]):.5f}",
                    "label": f"{float(labels_cpu[j]):.5f}",
                }

    avg_loss = total_loss / len(dataloader)
    print(f"\n[Test Evaluation] MSE Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Prediction results saved to {results_path}")

    if all_preds and all_labels:
        mse = mean_squared_error(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)
        print(f"\nTest Set Evaluation Results")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
    else:
        print("Failed to calculate MSE/MAE, no successful inferences available.")

    return avg_loss, all_preds, all_labels


MODES: Dict[str, Dict[str, str]] = {
    "full": {
        "model_dir": "./model_save/regression_lora_output_selected_AU_last_{model}/best_{q}",
        "test_file": "/root/autodl-tmp/Prolific/data_json/{q}_test_score_prediction.json",
        "results_path": "./results/regression_predictions_selected_AU_new/{q}_test_score_prediction_outputs.json",
    },
    "all_au": {
        "model_dir": "./model_save/ablation_all_AU_{model}/best_{q}",  # w/o AU selection
        "test_file": "/root/autodl-tmp/Prolific/ablation_data/ablation_all_AU/{q}_test_score_prediction.json",
        "results_path": "./results/ablation_all_AU/{q}_test_score_prediction_outputs.json",
    },
    "only_text": {
        "model_dir": "./model_save/ablation_only_text_{model}/best_{q}",  # w/o AU description
        "test_file": "/root/autodl-tmp/Prolific/ablation_data/ablation_only_text/{q}_test_score_prediction.json",
        "results_path": "./results/ablation_only_text/{q}_test_score_prediction_outputs.json",
    },
    "only_au": {
        "model_dir": "./model_save/ablation_only_AU_{model}/best_{q}",  # w/o text
        "test_file": "/root/autodl-tmp/Prolific/ablation_data/ablation_only_AU/{q}_test_score_prediction.json",
        "results_path": "./results/ablation_only_AU/{q}_test_score_prediction_outputs.json",
    },
}

MODE_ALIASES: Dict[str, str] = {
    "ablation_all_AU": "all_au",
    "ablation_only_text": "only_text",
    "ablation_only_AU": "only_au",
}

ALL_MODE_CHOICES = sorted(set(MODES.keys()) | set(MODE_ALIASES.keys()))


def parse_args():
    parser = argparse.ArgumentParser(description="Unified evaluation & ablation runner.")
    parser.add_argument("--mode", choices=ALL_MODE_CHOICES, default="full", help="Choose experiment variant.")
    parser.add_argument("--q", type=str, default="q3", help="Question id (e.g., q3).")
    parser.add_argument("--model-name", type=str, default="Llama3")
    parser.add_argument("--model-path", type=str, default="/root/autodl-tmp/LLM-Research/Meta-Llama-3___1-8B-Instruct")
    parser.add_argument("--model-dir", type=str, default=None, help="Override fine-tuned checkpoint dir (expects /best_{q}).")
    parser.add_argument("--test-file", type=str, default=None, help="Override test json file.")
    parser.add_argument("--results-path", type=str, default=None, help="Where to save prediction json.")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    mode = MODE_ALIASES.get(args.mode, args.mode)
    cfg = MODES[mode]
    model_dir = args.model_dir or cfg["model_dir"].format(model=args.model_name, q=args.q)
    test_file = args.test_file or cfg["test_file"].format(q=args.q)
    results_path = args.results_path or cfg["results_path"].format(q=args.q)

    print(f"Loading test data: {test_file}")
    test_samples = load_json_samples(test_file)
    convert_output_to_float(test_samples)

    print(f"Loading tokenizer: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {args.model_path}")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )

    print(f"Loading LoRA adapter: {model_dir}")
    lora_model = PeftModel.from_pretrained(base_model, model_dir)
    model = LLMWithRegressionHead(lora_model, tokenizer)

    regression_head_path = os.path.join(model_dir, "regression_head.pt")
    print(f"Loading regression head weights: {regression_head_path}")
    model.regression_head.load_state_dict(torch.load(regression_head_path, map_location="cpu"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    base_test_dataset = GeneralDataset(test_samples, tokenizer, formatting_func, args.context_length, add_special_tokens=True)
    test_dataset = ExtendedDataset(base_test_dataset, test_samples)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    evaluate(model, test_loader, device, args.q, results_path)


if __name__ == "__main__":
    main()
