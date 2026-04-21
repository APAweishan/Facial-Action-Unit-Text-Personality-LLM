import argparse
import os
from typing import List

import pandas as pd

from dataclasses import dataclass
from typing import Dict, Iterable, Optional


@dataclass(frozen=True)
class SplitSpec:
    """
    Split definition:
    splits[split_name] = list of ids (strings)
    """

    splits: Dict[str, List[str]]


def load_splits_from_single_csv(df, id_col: str, split_col: str, split_names: Iterable[str]) -> SplitSpec:
    if id_col not in df.columns or split_col not in df.columns:
        raise ValueError(f"CSV must contain '{id_col}' and '{split_col}' columns")

    splits: Dict[str, List[str]] = {}
    for split_name in split_names:
        splits[split_name] = df[df[split_col] == split_name][id_col].astype(str).tolist()
    return SplitSpec(splits=splits)


def load_splits_from_three_csvs(train_df, valid_df, test_df, id_col: str) -> SplitSpec:
    if id_col not in train_df.columns or id_col not in valid_df.columns or id_col not in test_df.columns:
        raise ValueError(f"CSV must contain '{id_col}' column")

    return SplitSpec(
        splits={
            "train": train_df[id_col].astype(str).tolist(),
            "valid": valid_df[id_col].astype(str).tolist(),
            "test": test_df[id_col].astype(str).tolist(),
        }
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def move_items_by_q_split(
    *,
    base_dir: str,
    src_root: str,
    splits: SplitSpec,
    questions: List[str],
    file_ext: str,
    src_name_template: str,
    dst_name_template: str,
    dst_folder_template: str,
    verbose_missing: bool = True,
) -> None:
    """
    Generic organizer that moves files from a flat folder into:
      dst_folder_template.format(base_dir=..., q=..., split=...)
    and renames:
      src_name_template.format(id=..., q=..., ext=...)
      dst_name_template.format(id=..., q=..., ext=...)

    Notes:
    - We keep shutil.move (original behavior) to preserve existing data pipelines.
    - dst folder template controls whether modality is included as a subdir.
    """

    import shutil

    for q in questions:
        for split_name, id_list in splits.splits.items():
            dst_folder = dst_folder_template.format(base_dir=base_dir, q=q, split=split_name)
            ensure_dir(dst_folder)

            for id_ in id_list:
                src_filename = src_name_template.format(id=id_, q=q, ext=file_ext)
                src_path = os.path.join(src_root, src_filename)
                dst_filename = dst_name_template.format(id=id_, q=q, ext=file_ext)
                dst_path = os.path.join(dst_folder, dst_filename)

                if os.path.exists(src_path):
                    shutil.move(src_path, dst_path)
                elif verbose_missing:
                    print(f"WARNING: file does not exist: {src_path}")


def parse_csv_list(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organize flat assets (text/videos) into base_dir/q/split[/modality]/ by moving files."
    )
    parser.add_argument("--base-dir", default="/root/autodl-tmp", help="Base directory containing CSVs and modality folder.")
    parser.add_argument("--modality", choices=["text", "videos"], default="text", help="Asset modality folder name.")
    parser.add_argument("--questions", default="q1,q2,q3,q4,q5,q6", help="Comma-separated questions list.")

    parser.add_argument("--train-csv", default="train_data.csv", help="Train split CSV (relative to base-dir).")
    parser.add_argument("--valid-csv", default="valid_data.csv", help="Valid split CSV (relative to base-dir).")
    parser.add_argument("--test-csv", default="test_data.csv", help="Test split CSV (relative to base-dir).")
    parser.add_argument("--id-col", default="id", help="ID column in split CSVs.")

    parser.add_argument("--file-ext", default=None, help="Override file extension (.txt/.mp4).")
    parser.add_argument(
        "--dst-includes-modality",
        action="store_true",
        help="Write into base_dir/q/split/modality/ (defaults to True for text).",
    )
    parser.add_argument(
        "--dst-excludes-modality",
        action="store_true",
        help="Write into base_dir/q/split/ (defaults to True for videos).",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    modality = args.modality
    questions = parse_csv_list(args.questions)

    file_ext = args.file_ext
    if not file_ext:
        file_ext = ".txt" if modality == "text" else ".mp4"

    train_df = pd.read_csv(os.path.join(base_dir, args.train_csv))
    valid_df = pd.read_csv(os.path.join(base_dir, args.valid_csv))
    test_df = pd.read_csv(os.path.join(base_dir, args.test_csv))
    splits = load_splits_from_three_csvs(train_df, valid_df, test_df, id_col=args.id_col)

    if args.dst_includes_modality and args.dst_excludes_modality:
        raise ValueError("Choose only one of --dst-includes-modality / --dst-excludes-modality")

    if args.dst_includes_modality:
        dst_includes_modality = True
    elif args.dst_excludes_modality:
        dst_includes_modality = False
    else:
        dst_includes_modality = modality == "text"

    src_root = os.path.join(base_dir, modality)
    if dst_includes_modality:
        dst_folder_template = os.path.join("{base_dir}", "{q}", "{split}", modality)
    else:
        dst_folder_template = os.path.join("{base_dir}", "{q}", "{split}")

    move_items_by_q_split(
        base_dir=base_dir,
        src_root=src_root,
        splits=splits,
        questions=questions,
        file_ext=file_ext,
        src_name_template="{id}_{q}{ext}",
        dst_name_template="{id}{ext}",
        dst_folder_template=dst_folder_template,
        verbose_missing=True,
    )


if __name__ == "__main__":
    main()
