import os
import json
import argparse
from typing import Dict, List

from tqdm import tqdm

# -----------------------------
# Shared maps
# -----------------------------
trait_map = {
    "q3": "Honesty-Humility",
    "q4": "Extraversion",
    "q5": "Agreeableness",
    "q6": "Conscientiousness",
}

question_map = {
    "q1": "What would you consider among your greatest strengths and weaknesses as an employee?",
    "q2": "How would your best friend describe you?",
    "q3": "Think of situations when you made professional decisions that could affect your status or how much money you make. How do you usually behave in such situations? Why do you think that is?",
    "q4": "Think of situations when you joined a new team of people. How do you usually behave when you enter a new team? Why do you think that is?",
    "q5": "Think of situations when someone annoyed you. How do you usually react in such situations? Why do you think that is?",
    "q6": "Think of situations when your work or workspace were not very organized. How typical is that of you? Why do you think that is?",
}

definition_map = {
    "q3": "Persons with very high scores on the Honesty-Humility scale avoid manipulating others for personal gain, feel little temptation to break rules, are uninterested in lavish wealth and luxuries, and feel no special entitlement to elevated social status. Conversely, persons with very low scores on this scale will flatter others to get what they want, are inclined to break rules for personal profit, are motivated by material gain, and feel a strong sense of self-importance.",
    "q4": "Persons with very high scores on the Extraversion scale feel positively about themselves, feel confident when leading or addressing groups of people, enjoy social gatherings and interactions, and experience positive feelings of enthusiasm and energy. Conversely, persons with very low scores on this scale consider themselves unpopular, feel awkward when they are the center of social attention, are indifferent to social activities, and feel less lively and optimistic than others do.",
    "q5": "Persons with very high scores on the Agreeableness scale forgive the wrongs that they suffered, are lenient in judging others, are willing to compromise and cooperate with others, and can easily control their temper. Conversely, persons with very low scores on this scale hold grudges against those who have harmed them, are rather critical of others' shortcomings, are stubborn in defending their point of view, and feel anger readily in response to mistreatment.",
    "q6": "Persons with very high scores on the Conscientiousness scale organize their time and their physical surroundings, work in a disciplined way toward their goals, strive for accuracy and perfection in their tasks, and deliberate carefully when making decisions. Conversely, persons with very low scores on this scale tend to be unconcerned with orderly surroundings or schedules, avoid difficult tasks or challenging goals, are satisfied with work that contains some errors, and make decisions on impulse or with little reflection.",
}

# -----------------------------
# Mode configs (keep defaults to match original scripts)
# -----------------------------
MODES: Dict[str, Dict] = {
    # Formal Method: selected AU + text
    "full": {
        "description_root": "/root/autodl-tmp/Prolific/raw_data/processed_descriptions_selected_AU-4_new",
        "answer_root": "/root/autodl-tmp/Prolific/raw_data/text_transcripts",
        "output_dir": "/root/autodl-tmp/Prolific/data_json",
        "template": "full",
    },
    # w/o AU selection: Use all AU descriptions + text
    "all_au": {
        "description_root": "/root/autodl-tmp/Prolific/raw_data/processed_descriptions_all_AUs",
        "answer_root": "/root/autodl-tmp/Prolific/raw_data/text_transcripts",
        "output_dir": "/root/autodl-tmp/Prolific/ablation_data/ablation_all_AU",
        "template": "full",
    },
    "only_au": {
        "description_root": "/root/autodl-tmp/Prolific/raw_data/processed_descriptions_selected_AU-4_new",
        "answer_root": None,
        "output_dir": "/root/autodl-tmp/Prolific/ablation_data/ablation_only_AU",  # w/o text
        "template": "au_only",
    },
    "only_text": {
        "description_root": None,
        "answer_root": "/root/autodl-tmp/Prolific/raw_data/text_transcripts",
        "output_dir": "/root/autodl-tmp/Prolific/ablation_data/ablation_only_text",  # w/o AU description
        "template": "text_only",
    },
}

# Backward-compatible aliases (keep old CLI values working)
MODE_ALIASES = {
    "only_AU": "only_au",
}

ALL_MODE_CHOICES = sorted(set(MODES.keys()) | set(MODE_ALIASES.keys()))


# -----------------------------
# Builders
# -----------------------------
def build_instruction(trait: str) -> str:
    return f"""
You are an expert in the HEXACO personality model. Your task is to analyze the {trait} subdimension and generate an embedding that captures personality-relevant patterns from the information and definition.
"""


def build_input(description: str, answer: str, question_text: str, trait: str, trait_definition: str, template: str) -> str:
    if template == "full":
        body = f"""
Information extracted from the video:
1.Semantic description of facial action units trends:{description}
2.Textual answers:{answer}
"""
    elif template == "au_only":
        body = f"""
Information extracted from the video:

1.Semantic description of facial action units trends:{description}
"""
    elif template == "text_only":
        body = f"""
Information extracted from the video:

1.Textual answers:{answer}
"""
    else:
        raise ValueError(f"Unknown template: {template}")

    return f"""
The subject was asked to record a video in response to the personality-eliciting question: "{question_text}". 

{body}

Definition of {trait} in HEXACO:
{trait_definition}
"""


# -----------------------------
# Core generation
# -----------------------------
def process_split(split: str, cfg: Dict, q_list: List[str]):
    label_files = cfg["label_files"]
    description_root = cfg["description_root"]
    answer_root = cfg["answer_root"]
    output_dir = cfg["output_dir"]
    template = cfg["template"]

    with open(label_files[split], "r", encoding="utf-8") as f:
        label_dict = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for q in q_list:
        trait = trait_map[q]
        question_text = question_map[q]
        trait_definition = definition_map[q]
        formatted_samples = {}

        description_dir = os.path.join(description_root, q, split) if description_root else None
        answer_dir = os.path.join(answer_root, q, split, "text") if answer_root else None

        if description_dir and not os.path.isdir(description_dir):
            print(f"❌ Description directory does not exist: {description_dir}")
            continue
        if answer_dir and not os.path.isdir(answer_dir):
            print(f"❌ Text directory does not exist: {answer_dir}")
            continue

        source_dir = description_dir or answer_dir
        for fname in tqdm(sorted(os.listdir(source_dir)), desc=f"{split}-{q}"):
            if not fname.endswith(".txt"):
                continue
            id_ = fname.replace(".txt", "")

            if id_ not in label_dict or trait not in label_dict[id_]:
                continue

            try:
                description = ""
                answer = ""
                if description_dir:
                    with open(os.path.join(description_dir, fname), encoding="utf-8") as f:
                        description = f.read().strip()
                if answer_dir:
                    with open(os.path.join(answer_dir, fname), encoding="utf-8") as f:
                        answer = f.read().strip()
                score = label_dict[id_][trait]
            except Exception as e:
                print(f"⚠️ Failed to read: {id_} | Error: {e}")
                continue

            instruction_text = build_instruction(trait)
            input_text = build_input(description, answer, question_text, trait, trait_definition, template)
            output_text = f"{score:.5f}"

            formatted_samples[id_] = {
                "instruction": instruction_text,
                "input": input_text,
                "output": output_text,
            }

        sorted_samples = dict(sorted(formatted_samples.items()))
        out_path = os.path.join(output_dir, f"{q}_{split}_score_prediction.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sorted_samples, f, indent=2, ensure_ascii=False)
        print(f"✅ Written to {out_path}, Sample Count: {len(formatted_samples)}")


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Unified data generator for full & ablation modalities.")
    parser.add_argument("--mode", choices=ALL_MODE_CHOICES, default="full", help="Select data generation variant.")
    parser.add_argument("--splits", default="train,valid,test", help="Comma list of splits.")
    parser.add_argument("--q-list", default="q3,q4,q5,q6", help="Comma list of q ids (applied to all modes).")
    parser.add_argument("--label-train", default="/root/autodl-tmp/Prolific/annotation_train.json")
    parser.add_argument("--label-valid", default="/root/autodl-tmp/Prolific/annotation_valid.json")
    parser.add_argument("--label-test", default="/root/autodl-tmp/Prolific/annotation_test.json")
    parser.add_argument("--description-root", default=None)
    parser.add_argument("--answer-root", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def parse_csv_list(csv: str) -> List[str]:
    return [item.strip() for item in csv.split(",") if item.strip()]


def main():
    args = parse_args()
    mode = MODE_ALIASES.get(args.mode, args.mode)
    cfg = MODES[mode].copy()

    # Shared label files (same across modes; overridable)
    cfg["label_files"] = {
        "train": args.label_train,
        "valid": args.label_valid,
        "test": args.label_test,
    }

    # Optional overrides
    if args.description_root:
        cfg["description_root"] = args.description_root
    if args.answer_root:
        cfg["answer_root"] = args.answer_root
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    splits = parse_csv_list(args.splits)
    q_list = parse_csv_list(args.q_list)

    for split in splits:
        process_split(split, cfg, q_list)


if __name__ == "__main__":
    main()
