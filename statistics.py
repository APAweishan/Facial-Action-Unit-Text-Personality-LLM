import argparse
import json
import os
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error


TRAIT_MAP = {
    "q3": "Honesty-Humility",
    "q4": "Extraversion",
    "q5": "Agreeableness",
    "q6": "Conscientiousness",
}


def parse_csv_list(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def load_predictions(file_path: str) -> Optional[tuple[list[float], list[float]]]:
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    preds = [float(item["pred"]) for item in data.values()]
    labels = [float(item["label"]) for item in data.values()]
    return preds, labels


def print_metrics(q: str, preds: Iterable[float], labels: Iterable[float]) -> None:
    preds = list(preds)
    labels = list(labels)
    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    rmse = float(np.sqrt(mse))
    pearson_corr, p_value = pearsonr(labels, preds)

    print(f"Metrics | {q} ({TRAIT_MAP.get(q, q)})")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"Pearson: {pearson_corr} (p={p_value})")
    print("-" * 50)


def plot_histogram(q: str, preds: List[float], labels: List[float], bins: int = 20, show: bool = True, save_path: str = "") -> None:
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    plt.hist(preds, bins=bins, alpha=0.7, label="Predictions", color="skyblue", edgecolor="black", density=True)
    plt.hist(labels, bins=bins, alpha=0.7, label="True Labels", color="orange", edgecolor="black", density=True)

    min_val = min(min(preds), min(labels))
    max_val = max(max(preds), max(labels))
    x_range = np.linspace(min_val, max_val, 100)

    mean_pred, std_pred = norm.fit(preds)
    mean_label, std_label = norm.fit(labels)

    plt.plot(x_range, norm.pdf(x_range, mean_pred, std_pred), "b-", lw=2, label="Fitted Normal - Predictions")
    plt.plot(x_range, norm.pdf(x_range, mean_label, std_label), "orange", lw=2, label="Fitted Normal - True Labels")

    plt.title(f"{TRAIT_MAP.get(q, q)} - Prediction vs True Labels", fontsize=14)
    plt.xlabel("Values", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute metrics and plot histograms from evaluate.py outputs.")
    parser.add_argument(
        "--folder-path",
        type=str,
        required=True,
        help="Folder containing q*_test_score_prediction_outputs.json.",
    )
    parser.add_argument("--q-list", type=str, default="q3,q4,q5,q6", help="Comma-separated q ids.")
    parser.add_argument("--bins", type=int, default=20, help="Histogram bins.")
    parser.add_argument("--no-show", action="store_true", help="Do not open matplotlib windows.")
    parser.add_argument("--save-dir", type=str, default="", help="If set, save figures into this directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    q_list = parse_csv_list(args.q_list)
    folder_path = args.folder_path

    missing = 0
    for q in q_list:
        file_name = f"{q}_test_score_prediction_outputs.json"
        file_path = os.path.join(folder_path, file_name)
        loaded = load_predictions(file_path)
        if loaded is None:
            print(f"Missing: {file_path}")
            missing += 1
            continue

        preds, labels = loaded
        print_metrics(q, preds, labels)

        save_path = ""
        if args.save_dir:
            save_path = os.path.join(args.save_dir, f"{q}_pred_vs_label_hist.png")
        plot_histogram(q, preds, labels, bins=args.bins, show=(not args.no_show), save_path=save_path)

    if missing == len(q_list):
        raise FileNotFoundError(f"No prediction json files found under: {folder_path}")


if __name__ == "__main__":
    main()

