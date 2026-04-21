import os
import re
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from typing import Callable, List

# -----------------------------
# Shared AU metadata & examples
# -----------------------------

ALL_AUS = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
]

AU_SUBSETS = {
    "q3": ["AU06_r", "AU09_r", "AU12_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU45_r"],
    "q4": ["AU02_r", "AU05_r", "AU06_r", "AU07_r", "AU12_r", "AU14_r", "AU17_r", "AU23_r", "AU45_r"],
    "q5": ["AU05_r", "AU06_r", "AU12_r", "AU14_r", "AU15_r", "AU20_r", "AU26_r"],
    "q6": ["AU02_r", "AU10_r", "AU12_r", "AU20_r", "AU23_r", "AU26_r"],
}

au_descriptions = {
    "AU01": "The inner corners of the eyebrows are lifted slightly, the skin of the glabella and forehead above it is lifted slightly and wrinkles deepen slightly and a trace of new ones form in the center of the forehead.",
    "AU02": "The outer part of the eyebrow raise is pronounced. The wrinkling above the right outer eyebrow has increased markedly, and the wrinkling on the left is pronounced.",
    "AU04": "Vertical wrinkles appear in the glabella and the eyebrows are pulled together. The inner parts of the eye-brows are pulled down a trace on the right and slightly on the left with traces of wrinkling at the corners.",
    "AU05": "The upper lip is raised vertically, exposing more of the teeth and gums slightly. The skin above the lip stretches upward, creating subtle wrinkles or tension lines between the nose and the upper lip.",
    "AU06": "The cheeks are lifted without raising the lip corners. The infraorbital furrow has deepened slightly, and wrinkles under the eyes increase.",
    "AU07": "The lower eyelid is raised markedly, causing bulging and narrowing of the eye aperture.",
    "AU09": "Wrinkling the nose and lifting the nasal wings, which deepens the upper nasolabial fold as the upper lip is drawn up.",
    "AU10": "The upper lip is drawn straight up, the outer portions of the lip are raised slightly.",
    "AU12": "The corners of the lips are raised obliquely.",
    "AU14": "The lip corners are tightly pulled inward, causing significant wrinkling around the corners and stretching of the skin on the chin and lower lip.",
    "AU15": "The corners of the lips are pulled down slightly, with some lateral pulling.",
    "AU17": "Severe wrinkling of the chin with the lower lip pushed up and out.",
    "AU20": "The corners of the lips are pulled outward horizontally, the skin around the mouth is stretched thinly, and there is a noticeable tension along the line extending from the corners of the lips.",
    "AU23": "The lips are tightly pressed, narrowing the red parts and causing significant wrinkling around the lips.",
    "AU25": "The teeth are visible with lips slightly parted.",
    "AU26": "The jaw drops as much as possible, parting the lips.",
    "AU45": "The eyelids close abruptly for a brief moment, with the upper eyelid dropping sharply and the lower eyelid lifting slightly, causing the eye aperture to disappear temporarily before quickly reopening.",
}

EXAMPLE_ALL = """
Input (AU data):
AU measurements for frames 181 to 187 (AU intensities):

Frame | AU01 | AU02 | AU04 | AU05 | AU06 | AU07 | AU09 | AU10 | AU12 | AU14 | AU15 | AU17 | AU20 | AU23 | AU25 | AU26 | AU45
181   | 0.51 | 0.15 | 2.08 | 0.00 | 0.10 | 0.00 | 0.00 | 0.90 | 0.00 | 0.00 | 0.02 | 0.01 | 0.10 | 0.00 | 0.89 | 1.01 | 0.00
182   | 0.32 | 0.13 | 1.95 | 0.00 | 0.12 | 0.00 | 0.00 | 0.95 | 0.00 | 0.00 | 0.08 | 0.01 | 0.15 | 0.00 | 0.66 | 0.76 | 0.00
183   | 0.05 | 0.00 | 1.86 | 0.00 | 0.21 | 0.00 | 0.02 | 0.95 | 0.00 | 0.00 | 0.20 | 0.14 | 0.16 | 0.00 | 0.44 | 0.46 | 0.00
184   | 0.00 | 0.00 | 1.87 | 0.00 | 0.21 | 0.03 | 0.03 | 0.89 | 0.00 | 0.00 | 0.26 | 0.24 | 0.15 | 0.00 | 0.47 | 0.27 | 0.00
185   | 0.00 | 0.00 | 1.93 | 0.00 | 0.24 | 0.07 | 0.09 | 0.76 | 0.01 | 0.00 | 0.20 | 0.32 | 0.14 | 0.00 | 0.62 | 0.10 | 0.00
186   | 0.00 | 0.00 | 1.84 | 0.00 | 0.19 | 0.22 | 0.11 | 0.60 | 0.02 | 0.00 | 0.09 | 0.32 | 0.09 | 0.00 | 0.82 | 0.03 | 0.00
187   | 0.00 | 0.00 | 1.74 | 0.00 | 0.16 | 0.22 | 0.13 | 0.45 | 0.02 | 0.00 | 0.00 | 0.34 | 0.04 | 0.00 | 0.97 | 0.00 | 0.00
Output (Description):
From frames 181 to 187, the inner corners of the eyebrows begin slightly lifted but gradually lower until they remain neutral, while the outer parts of the eyebrows start with a faint rise that quickly fades and stays unchanged. The eyebrows are consistently drawn together, maintaining visible vertical wrinkling between them with only minor softening toward the end. The upper lip shows a subtle upward pull at first, exposing the teeth slightly, but this weakens steadily. The cheeks are slightly raised with faint deepening of the lines beneath the eyes, while the lower eyelids become increasingly tightened, narrowing the eye aperture. The nose shows a faint upward wrinkling and lifting of the nasal wings, which grows slightly more pronounced across the sequence. The upper lip is drawn upward throughout, though the effect diminishes as the frames progress. The lip corners show little movement overall, with only minimal upward lift or tightening briefly appearing, then fading. There is no notable inward tightening of the lip corners, while the lips are drawn slightly downward with a subtle pushing up of the lower lip that becomes more evident toward the end, accompanied by slight chin wrinkling. At the same time, the corners of the lips are pulled outward horizontally with mild tension that gradually decreases, and the lips remain mostly relaxed without tight pressing or bulging. The lips part slightly, showing intermittent visibility of the teeth, while the jaw lowers modestly at the beginning and gradually closes. The eyelids remain open throughout with no abrupt closure.
"""

EXAMPLES_DICT = {
    "q3": {
        "window_data": """
AU measurements for frames 146 to 152 (AU intensities):

Frame | AU06 | AU09 | AU12 | AU17 | AU20 | AU23 | AU25 | AU45
146   | 1.84 | 0.05 | 0.04 | 0.00 | 0.00 | 0.06 | 0.71 | 0.25
147   | 1.70 | 0.05 | 0.03 | 0.08 | 0.06 | 0.16 | 0.56 | 0.24
148   | 1.63 | 0.05 | 0.03 | 0.31 | 0.06 | 0.31 | 0.33 | 0.34
149   | 1.62 | 0.00 | 0.03 | 0.46 | 0.00 | 0.35 | 0.00 | 0.37
150   | 1.67 | 0.00 | 0.00 | 0.48 | 0.00 | 0.19 | 0.00 | 0.42
151   | 1.76 | 0.04 | 0.03 | 0.25 | 0.00 | 0.04 | 0.00 | 0.39
152   | 1.85 | 0.04 | 0.10 | 0.11 | 0.12 | 0.00 | 0.24 | 0.31
""",
        "description": """
From frames 146 to 152, the cheeks remain gently lifted, sustaining a consistent upward presence throughout the sequence. Subtle wrinkling near the nose appears briefly in the early frames but remains faint. The corners of the lips show only slight upward movement at times, while the chin pushes forward and wrinkles deepen midway before gradually relaxing again. Around the mouth, the lips fluctuate between mild pressing and release, occasionally stretching outward with faint tension. The lips part slightly at the start, then close, and reopen toward the end. Several brief eye closures occur, each marked by a quick narrowing of the eyes before reopening.
""",
    },
    "q4": {
        "window_data": """
AU measurements for frames 201 to 207 (AU intensities):

Frame | AU02 | AU05 | AU06 | AU07 | AU12 | AU14 | AU17 | AU23 | AU45
201   | 0.00 | 0.00 | 0.00 | 0.82 | 0.54 | 0.36 | 0.19 | 0.96 | 0.00
202   | 0.00 | 0.00 | 0.00 | 1.18 | 0.62 | 0.30 | 0.11 | 0.69 | 0.00
203   | 0.00 | 0.00 | 0.00 | 1.38 | 0.65 | 0.23 | 0.07 | 0.52 | 0.00
204   | 0.00 | 0.00 | 0.00 | 1.50 | 0.65 | 0.25 | 0.08 | 0.45 | 0.00
205   | 0.00 | 0.00 | 0.00 | 1.53 | 0.63 | 0.18 | 0.16 | 0.42 | 0.00
206   | 0.00 | 0.00 | 0.00 | 1.48 | 0.64 | 0.09 | 0.23 | 0.38 | 0.00
207   | 0.00 | 0.00 | 0.00 | 1.31 | 0.57 | 0.11 | 0.34 | 0.43 | 0.00
""",
        "description": """
From frames 201 to 207, the eyes remain open without signs of blinking, while the lower eyelids rise steadily, causing the eyes to narrow progressively before easing slightly toward the end. The corners of the lips show a subtle upward pull that peaks in the middle frames before softening again. Around the mouth, there is clear pressing of the lips that is most pronounced at the start and gradually decreases in intensity as the sequence unfolds. The chin shows faint forward movement with mild wrinkling that becomes more noticeable in the later frames. Overall, the expression conveys sustained eyelid tension paired with gentle mouth activity that transitions from strong compression toward a more relaxed state.
""",
    },
    "q5": {
        "window_data": """
AU measurements for frames 102 to 108 (AU intensities):

Frame | AU05 | AU06 | AU12 | AU14 | AU15 | AU20 | AU26
102   | 0.00 | 0.00 | 0.61 | 0.00 | 0.57 | 0.16 | 0.04
103   | 0.00 | 0.00 | 0.58 | 0.00 | 0.52 | 0.19 | 0.04
104   | 0.00 | 0.00 | 0.62 | 0.00 | 0.53 | 0.13 | 0.04
105   | 0.00 | 0.00 | 0.55 | 0.00 | 0.42 | 0.25 | 0.00
106   | 0.13 | 0.00 | 0.40 | 0.00 | 0.29 | 0.18 | 0.16
107   | 0.13 | 0.00 | 0.27 | 0.00 | 0.30 | 0.21 | 0.33
108   | 0.17 | 0.00 | 0.06 | 0.00 | 0.36 | 0.10 | 0.33
""",
        "description": """
From frames 102 to 108, the corners of the lips begin with a gentle upward pull that steadily diminishes, fading almost completely by the final frames. At the same time, the lips show a light outward stretch that grows more noticeable midway before easing again. The lower face gradually reduces its downward pull, softening the overall tension, while the lips transition from being mostly closed to slightly parted toward the end. The cheeks remain inactive throughout, leaving the primary movements centered around the mouth and chin area, with the sequence overall reflecting a slow release of lip elevation into mild stretching and parting.
""",
    },
    "q6": {
        "window_data": """
AU measurements for frames 87 to 93 (AU intensities):

Frame | AU02 | AU10 | AU12 | AU20 | AU23 | AU26
87    | 0.00 | 0.13 | 0.89 | 0.04 | 0.35 | 0.19
88    | 0.00 | 0.10 | 0.96 | 0.03 | 0.51 | 0.30
89    | 0.00 | 0.11 | 1.05 | 0.02 | 0.60 | 0.49
90    | 0.00 | 0.06 | 1.03 | 0.00 | 0.68 | 0.56
91    | 0.00 | 0.01 | 0.95 | 0.00 | 0.62 | 0.55
92    | 0.00 | 0.00 | 0.83 | 0.00 | 0.72 | 0.60
93    | 0.00 | 0.00 | 0.80 | 0.00 | 0.75 | 0.75
""",
        "description": """
From frames 87 to 93, the corners of the lips rise into a noticeable upward pull that peaks early and then gradually eases. Meanwhile, the lips press together with steadily increasing firmness, narrowing the shape as the sequence progresses. A slight outward stretch is visible at the very beginning but quickly disappears. The lips part modestly in the first half of the sequence, the opening becoming somewhat larger around the midpoint before closing again by the final frame. Overall, the movement is characterized by a brief lift of the mouth corners, progressive lip compression, and a controlled cycle of parting and reclosure.
""",
    },
}


# -----------------------------
# Helpers for mode differences
# -----------------------------

def get_example_for_q(q: str) -> str:
    example = EXAMPLES_DICT.get(q)
    if example is None:
        return "Example not found for the given q"
    return (
        "Input (AU Data):\n"
        f"{example['window_data']}\n"
        "Output (Description):\n"
        f"{example['description']}"
    )


def get_selected_columns(q: str) -> List[str]:
    return ["frame"] + AU_SUBSETS.get(q, [])


def _canonical_au_label(raw_code: str) -> str:
    """Normalize AU code (drop _r and leading zeros)."""
    base = raw_code.replace("_r", "")
    prefix, num = base[:2], base[2:]
    num = num.lstrip("0") or "0"
    return prefix + num


def format_dataframe_as_table_string(csv_path: str, selected_columns: List[str], double_newline: bool) -> str:
    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]
    df = df[[col for col in selected_columns if col in df.columns]]

    if df.empty:
        return "AU measurements table is empty."

    start_frame = int(df["frame"].min())
    end_frame = int(df["frame"].max())

    au_columns = [col.replace("_r", "") for col in df.columns if col != "frame"]
    header = "Frame | " + " | ".join(au_columns)

    table_lines = [header]
    for _, row in df.iterrows():
        values = [f"{int(row['frame'])}"] + [f"{row[col]:.2f}" for col in df.columns if col != "frame"]
        table_lines.append(" | ".join(values))

    sep = "\n\n" if double_newline else "\n"
    table_str = (
        f"AU measurements for frames {start_frame} to {end_frame} (AU intensities):{sep}"
        + "\n".join(table_lines)
    )
    return table_str.strip()


def generate_prompt_selected(q: str, window_data: str) -> str:
    selected_aus = AU_SUBSETS.get(q, [])
    relevant_aus = {}
    for au in selected_aus:
        key = _canonical_au_label(au)
        if key in au_descriptions:
            relevant_aus[key] = au_descriptions[key]
    semantic_descriptions = "\n".join([f"{au}: {desc}" for au, desc in relevant_aus.items()])
    focused_aus = ", ".join([_canonical_au_label(au) for au in selected_aus])
    example = get_example_for_q(q)

    prompt = f"""
You are currently acting as an expert in describing facial action dynamics based on Action Unit (AU) data.

Your goal is to generate a succinct, natural-language summary of the facial movement trends covering the entire duration of the specified AUs.

Here is a list of AUs you need to describe: {focused_aus}.

Below are semantic descriptions of the relevant AUs for your reference:

{semantic_descriptions}

🥥 Please follow these instructions carefully:
- Do NOT mention any AU identifiers (e.g., AU06, AU12) or numeric intensity values.
- Begin your description with the frame range in this exact format: "From frames XX to YY,".
- Summarize only physical AU activity and observable trends during the frames.
- Avoid referencing any AUs not in the specified list.
- Write a single, continuous paragraph without splitting into multiple paragraphs.
- Do NOT include field names such as "Input:", "Output:", or any dictionary/JSON syntax.
- Refer only to the example below for style and content.
- The example includes an AU data table followed by a natural language description.
- Use a similar style but avoid numeric data replication.
- If any AU data is missing or ambiguous, describe only what is clearly observable from the data provided.
- Strictly DO NOT describe or mention any facial movements or muscle actions that are NOT part of the provided AU list. 
If you notice any movement that is not included, completely omit it from your description. Failure to comply will result in an invalid output.

Example:
{example}

Now, based on this example, please summarize the facial actions for the given frame range:
Input (AU Data):
{window_data}
Output (Description):
"""
    return prompt


def generate_prompt_all(q: str, window_data: str) -> str:
    semantic_descriptions = "\n".join([f"{au}: {desc}" for au, desc in au_descriptions.items()])
    prompt = f"""
You are currently acting as an expert in describing facial action dynamics based on Action Unit (AU) data.

Your goal is to generate a succinct, natural-language summary of the facial movement trends covering the entire duration of the specified AUs.

Below are semantic descriptions of the relevant AUs for your reference:

{semantic_descriptions}

🥥 Please follow these instructions carefully:
- Do NOT mention any AU identifiers (e.g., AU06, AU12) or numeric intensity values.
- Begin your description with the frame range in this exact format: "From frames XX to YY,".
- Summarize only physical AU activity and observable trends during the frames.
- Include every AU in the provided list, even if it shows minimal or no change (describe as stable/unchanged if needed).
- Write a single, continuous paragraph without splitting into multiple paragraphs.
- Do NOT include field names such as "Input:", "Output:", or any dictionary/JSON syntax.
- Refer only to the example below for style and content.
- The example includes an AU data table followed by a natural language description.
- Use a similar style but avoid numeric data replication.
If you notice any movement that is not included, completely omit it from your description. Failure to comply will result in an invalid output.

Example:
{EXAMPLE_ALL}

Now, based on this example, please summarize the facial actions for the given frame range:
{window_data}
"""
    return prompt


# -----------------------------
# Processing pipeline (shared)
# -----------------------------

@dataclass
class ModeConfig:
    au_selector: Callable[[str, str], List[str]]
    prompt_builder: Callable[..., str]
    double_newline: bool


MODES = {
    "selected": ModeConfig(
        au_selector=get_selected_columns,
        prompt_builder=lambda q, window_data: generate_prompt_selected(q, window_data),
        double_newline=True,
    ),
    "all": ModeConfig(
        au_selector=lambda q: ["frame"] + ALL_AUS,
        prompt_builder=lambda q, window_data: generate_prompt_all(q, window_data),
        double_newline=False,
    ),
}


def get_mode_config(mode: str) -> ModeConfig:
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}'. Available: {list(MODES.keys())}")
    return MODES[mode]


def call_llm(client, prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def process_video_folder(video_folder: str, descriptions_output_folder: str, q: str, mode_cfg: ModeConfig, client) -> None:
    all_descriptions = []
    video_id = os.path.basename(video_folder)

    os.makedirs(descriptions_output_folder, exist_ok=True)
    output_txt = os.path.join(descriptions_output_folder, f"{video_id}.txt")
    if os.path.exists(output_txt) and os.path.getsize(output_txt) > 0:
        print(f"⚠️ Skip processed video: {video_id}")
        return

    subfolders = [f for f in os.listdir(video_folder) if f.startswith("small_window_")]
    subfolders.sort(key=lambda x: int(x.split("_")[-1]))

    for subfolder in subfolders:
        window_dir = os.path.join(video_folder, subfolder)
        keyframe_id = subfolder.split("_")[-1]
        csv_path = os.path.join(window_dir, f"frame_window_{keyframe_id}.csv")

        if not os.path.exists(csv_path):
            print(f"❌ Missing CSV file: {csv_path}")
            continue

        selected_cols = mode_cfg.au_selector(q)
        window_data = format_dataframe_as_table_string(csv_path, selected_cols, double_newline=mode_cfg.double_newline)
        prompt = mode_cfg.prompt_builder(q, window_data)

        description = call_llm(client, prompt)
        all_descriptions.append(description)

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_descriptions))
    print(f"✅ Description file has been saved: {output_txt}")


def process_dataset_folder(dataset_folder_path: str, descriptions_root: str, qx_folder_name: str, dataset_folder_name: str, q: str, mode_cfg: ModeConfig, client) -> None:
    video_items = [
        item
        for item in os.listdir(dataset_folder_path)
        if os.path.isdir(os.path.join(dataset_folder_path, item)) and re.match(r"^[a-zA-Z0-9_]+$", item)
    ]

    for item in tqdm(video_items, desc=f"Processing | q={qx_folder_name}/{dataset_folder_name}", ncols=100):
        video_folder = os.path.join(dataset_folder_path, item)
        print(f"\n🎬 Video being processed: {item}, location: {qx_folder_name}/{dataset_folder_name}")
        folder_shortname = dataset_folder_name.replace("_results", "")
        descriptions_output_folder = os.path.join(descriptions_root, qx_folder_name, folder_shortname)
        process_video_folder(video_folder, descriptions_output_folder, q, mode_cfg, client)


def batch_process_all(fau_root: str, qx_list: list, dataset_list: list, descriptions_root: str, mode: str, client) -> None:
    mode_cfg = get_mode_config(mode)

    for qx in qx_list:
        qx_path = os.path.join(fau_root, qx)
        if not os.path.isdir(qx_path):
            print(f"❌ Directory does not exist: {qx_path}")
            continue

        for dataset in dataset_list:
            dataset_path = os.path.join(qx_path, dataset)
            if not os.path.isdir(dataset_path):
                print(f"❌ Dataset directory missing: {dataset_path}")
                continue

            print(f"\n=== Starting processing {qx}/{dataset} ===")
            q = qx  
            process_dataset_folder(dataset_path, descriptions_root, qx, dataset, q, mode_cfg, client)
