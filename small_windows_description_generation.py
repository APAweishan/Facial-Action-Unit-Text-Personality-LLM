import argparse
import os
from typing import List

from openai import OpenAI

from description_core import batch_process_all


def parse_csv_list(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def build_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def main():
    parser = argparse.ArgumentParser(description="Generate AU-based descriptions via LLM.")
    parser.add_argument("--mode", choices=["selected", "all"], required=True, help="Experiment mode.")
    parser.add_argument("--fau-root", default=None, help="Root directory containing FAU data organized by qx and dataset folders.")
    parser.add_argument("--qx-list", default="q3", help="Comma-separated qx list, e.g., q1,q3.")
    parser.add_argument("--datasets", default="train_results,valid_results,test_results", help="Comma-separated dataset folders.")
    parser.add_argument("--descriptions-root", default="descriptions_all_AUs", help="Output root for generated descriptions.")
    parser.add_argument("--api-key", default="", help="OpenAI-compatible API key.")
    parser.add_argument("--base-url", default="", help="OpenAI-compatible base URL.")
    args = parser.parse_args()

    fau_root = args.fau_root
    qx_list = parse_csv_list(args.qx_list)
    dataset_list = parse_csv_list(args.datasets)

    client = build_client(args.api_key, args.base_url)

    batch_process_all(
        fau_root=fau_root,
        qx_list=qx_list,
        dataset_list=dataset_list,
        descriptions_root=args.descriptions_root,
        mode=args.mode,
        client=client,
    )


if __name__ == "__main__":
    main()
