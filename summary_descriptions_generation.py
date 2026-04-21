import argparse
import os
from typing import List

from openai import OpenAI
from tqdm import tqdm

from description_summary_core import (
    build_openai_client,
    make_llm_caller,
    make_prompt_for_mode,
    process_tree,
)


DEFAULTS_BY_MODE = {
    "selected": {
        "description_root": "/root/autodl-tmp/Prolific/descriptions_selected_AU",
        "output_root": "/root/autodl-tmp/Prolific/processed_descriptions_selected_AU",
    },
    "all": {
        "description_root": "/root/autodl-tmp/Prolific/descriptions_all_AUs",
        "output_root": "/root/autodl-tmp/Prolific/processed_descriptions_all_AUs",
    },
}


def parse_csv_list(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-window AU descriptions into one summary via LLM.")
    parser.add_argument("--mode", choices=["selected", "all"], required=True, help="Experiment mode.")

    parser.add_argument("--qx-list", default="q3,q4,q5,q6", help="Comma-separated q list, e.g., q3,q4.")
    parser.add_argument("--datasets", default="train,valid,test", help="Comma-separated dataset folders (splits).")

    parser.add_argument("--description-root", default=None, help="Input root containing q/split/*.txt.")
    parser.add_argument("--output-root", default=None, help="Output root to write q/split/*.txt.")

    parser.add_argument("--model", default="gpt-4o-2024-11-20", help="OpenAI model name.")
    parser.add_argument("--api-key", default="", help="OpenAI-compatible API key.")
    parser.add_argument("--base-url", default="", help="OpenAI-compatible base URL.")
    args = parser.parse_args()

    q_list = parse_csv_list(args.qx_list)
    split_list = parse_csv_list(args.datasets)

    defaults = DEFAULTS_BY_MODE[args.mode]
    description_root = args.description_root or defaults["description_root"]
    output_root = args.output_root or defaults["output_root"]

    client = build_openai_client(OpenAI, api_key=args.api_key, base_url=args.base_url)
    llm_call = make_llm_caller(client, model=args.model)

    def _make_prompt(description1: str, description2: str, q: str) -> str:
        return make_prompt_for_mode(description1, description2, q=q, mode=args.mode)

    if not os.path.isdir(description_root):
        raise FileNotFoundError(f"description_root does not exist or is not a directory: {description_root}")

    os.makedirs(output_root, exist_ok=True)
    process_tree(
        description_root=description_root,
        output_root=output_root,
        q_list=q_list,
        split_list=split_list,
        make_prompt=_make_prompt,
        llm_call=llm_call,
        tqdm=tqdm,
    )


if __name__ == "__main__":
    main()

