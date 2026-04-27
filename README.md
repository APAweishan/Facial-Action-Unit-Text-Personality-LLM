# LLM-based Multimodal Personality Recognition via Facial Action Unit-Text Semantic Fusion

This repository contains a **reproducible end-to-end pipeline** that predicts HEXACO personality trait scores (q3–q6) from video-based interview responses by combining:

- **OpenFace** facial Action Unit (AU) time series extraction,
- **Keyframe-centered small-window sampling** (non-continuous windows around salient motion peaks),
- **LLM-based semantic description generation** (per-window) + **LLM-based summary merging** (across windows),
- **LLM fine-tuning with a regression head** (LoRA + 4-bit quantization) to predict trait scores.

It is organized as a “paper-style” codebase: scripts are explicit, inputs/outputs are file-based, and ablations are supported via unified runners.

> If you use this code for research, please cite our paper (see [Citation](#8-citation)).

---

## 1. Project Overview

**Task.** Given a short video response to a personality-eliciting question, predict a continuous HEXACO trait score (Honesty-Humility / Extraversion / Agreeableness / Conscientiousness) per question `q3–q6`.

**Core idea.** Instead of directly modeling dense AU sequences, we:
1) detect salient moments and extract **small, sparse AU windows**,  
2) convert AU windows into **natural-language descriptions** using an LLM,  
3) merge window descriptions into a **single summary** per video (temporal trend approximation),  
4) train an LLM (with a regression head) on `{instruction, input → score}` JSON.

---

## 2. Repository Structure

```
.
├─ requirements.txt
│
├─ division_tool.py
├─ get_command_tool.py
├─ json_convertion_tool.py
│
├─ KeyFrameExtract.py
├─ au_selection.py
├─ description_core.py
├─ small_windows_description_generation.py
├─ description_summary_core.py
├─ summary_descriptions_generation.py
├─ generate_data_json.py
│
├─ dataset_module.py
├─ model_module.py
├─ train.py
├─ evaluate.py
├─ statistics.py
```

**One-line module responsibilities (verified from code).**

- `division_tool.py`: moves flat `text/` or `videos/` assets into `base_dir/q/split[/text]/` according to split CSVs.
- `get_command_tool.py`: prints **batched OpenFace** Windows commands (`FeatureExtraction.exe ... -aus`) for a folder of `.mp4`.
- `json_convertion_tool.py`: converts a label table (CSV) into JSON (despite the name mentioning Excel).
- `KeyFrameExtract.py`: from `(video.mp4, AU csv)` → extracts `small_window_*` folders with frames and `frame_window_*.csv`.
- `au_selection.py`: **simulated annealing** search over AU subsets using a simple LSTM baseline (standalone tool; not wired into the main pipeline).
- `description_core.py`: core logic for per-window AU-table → LLM prompt → description generation (called by `small_windows_description_generation.py`).
- `small_windows_description_generation.py`: CLI wrapper to generate **per-window descriptions** for each video folder under `*_results/`.
- `description_summary_core.py`: prompt + merge logic to iteratively merge multiple window descriptions into one summary.
- `summary_descriptions_generation.py`: CLI wrapper to generate **summary descriptions** per video (`q/split/*.txt` → `q/split/*.txt`).
- `generate_data_json.py`: builds `{instruction,input,output}` JSON for training/eval (supports full + ablation variants).
- `dataset_module.py`: tokenization + dataset wrappers using `tokenizer.apply_chat_template`.
- `model_module.py`: regression head on top of an LLM backbone (LoRA adapters are expected).
- `train.py`: unified LoRA+4bit training runner for full/ablation modes (saves `best_{q}` + `regression_head.pt`).
- `evaluate.py`: unified evaluation runner (loads base model + LoRA adapter + regression head; writes per-id predictions).
- `statistics.py`: computes metrics and plots histograms (non-CLI; edit `folder_path` to your results).

---

## 3. Environment Setup

### 3.1 Python

- Recommended: Python `>= 3.10`

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Notes:
- `torch==2.8.0+cu128` is pinned in `requirements.txt`. You may need to adjust it to match your CUDA / OS.
- For training/eval you need a GPU environment compatible with `bitsandbytes` 4-bit loading.

### 3.2 OpenFace (AU extraction)

This codebase assumes you have already extracted per-video AU CSV files using OpenFace **FeatureExtraction**.

- OpenFace project (install/build instructions): https://github.com/TadasBaltrusaitis/OpenFace

**Required CSV format (hard requirement from code):**
- one CSV per video: `.../<id>.csv`
- contains a `frame` column
- contains AU intensity columns named like: `AU01_r ... AU45_r` (see `description_core.ALL_AUS`)

---

## 4. Data Preparation

This pipeline is file-driven. The following directory conventions are **required by the scripts**.

### 4.1 Split definition CSVs (for `division_tool.py`)

You need three CSV files under `base_dir`:

- `train_data.csv`
- `valid_data.csv`
- `test_data.csv`

Each CSV must contain at least an ID column (default name: `id`), and IDs must match your filenames.

### 4.2 Raw assets (before running `division_tool.py`)

Place raw files in a flat structure under `base_dir`:

```
base_dir/
  videos/
    <id>_q3.mp4
    <id>_q4.mp4
    ...
  text/
    <id>_q3.txt
    <id>_q4.txt
    ...
  train_data.csv
  valid_data.csv
  test_data.csv
```

### 4.3 After re-organization (`division_tool.py` output)

Run twice (once for `videos`, once for `text`):

```bash
# videos: moves videos/ -> base_dir/qx/{train,valid,test}/<id>.mp4
python division_tool.py --base-dir <base_dir> --modality videos --questions q3,q4,q5,q6

# text: moves text/ -> base_dir/qx/{train,valid,test}/text/<id>.txt
python division_tool.py --base-dir <base_dir> --modality text --questions q3,q4,q5,q6
```

Resulting structure:

```
base_dir/
  q3/
    train/  <id>.mp4
    valid/  <id>.mp4
    test/   <id>.mp4
    train/text/  <id>.txt
    valid/text/  <id>.txt
    test/text/   <id>.txt
  q4/
  q5/
  q6/
```

Important:
- `division_tool.py` uses `shutil.move` (it **moves**, not copies). Use a backup if needed.
- Although `division_tool.py` supports `q1,q2`, the **core pipeline** (AU selection, description prompts, training defaults) is implemented for `q3–q6`.

### 4.4 OpenFace output paths (required by `KeyFrameExtract.py`)

For each `q` and `split`, place AU CSV files under:

```
<base_root>/<q>/<split>_csv/<id>.csv
```

Example:

```
base_dir/q3/train_csv/000123.csv
base_dir/q3/valid_csv/000123.csv
base_dir/q3/test_csv/000123.csv
```

---

## 5. Pipeline Usage (Most Important)

Below is the **real executable pipeline** according to the current codebase.

### Step 1: Data re-organization

**Input**
- `base_dir/videos/<id>_<q>.mp4`
- `base_dir/text/<id>_<q>.txt`
- `base_dir/{train,valid,test}_data.csv` with `id` column

**Output**
- `base_dir/<q>/{train,valid,test}/<id>.mp4`
- `base_dir/<q>/{train,valid,test}/text/<id>.txt`

**Script**
- `division_tool.py` (CLI)

**Command**

```bash
python division_tool.py --base-dir <base_dir> --modality videos --questions q3,q4,q5,q6
python division_tool.py --base-dir <base_dir> --modality text   --questions q3,q4,q5,q6
```

---

### Step 2: OpenFace AU extraction (external)

**Input**
- `<base_dir>/<q>/<split>/<id>.mp4`

**Output (required for next step)**
- `<base_dir>/<q>/<split>_csv/<id>.csv`

**Tool**
- OpenFace `FeatureExtraction` (external)

**Windows helper**
- `get_command_tool.py` prints batch commands for a folder of `.mp4`.
  - You must set `video_dir = r"..."` in the script before running.
  - The printed command currently does **not** include output directory options; ensure the resulting CSV files land in `<base_dir>/<q>/<split>_csv/`.

---

### Step 3: Keyframe-centered small-window extraction

**Input**
- videos: `<base_root>/<q>/<split>/<id>.mp4`
- AU csv: `<base_root>/<q>/<split>_csv/<id>.csv`

**Output**
- `<base_root>/<q>/<split>_results/<id>/small_window_<center_frame>/frame_window_<center_frame>.csv`

**Script**
- `KeyFrameExtract.py` (CLI)

**Command**

```bash
python KeyFrameExtract.py --base-root <base_root> --window-size 7
```

Notes (code behavior):
- The script processes `q in {q3,q4,q5,q6}` and `split in {train,valid,test}` (hard-coded).
- It samples the video every `10` frames (`key_frame_interval=10`) and finds local maxima on smoothed frame-diff signals.
- If `<id>/` already contains any `small_window_*`, it is skipped (no overwrite).

---

### Step 4: AU selection (simulated annealing)

**Status in this repo**
- The “selected AU lists” used in description prompts are currently **hard-coded** in `description_core.AU_SUBSETS`.
- `au_selection.py` is a standalone tool to search AU subsets using simulated annealing + an LSTM baseline.

**Input expected by `au_selection.py`**
- a directory layout like:

```
<data_dir>/
  train_data.csv
  valid_data.csv
  test_data.csv
  FAU_csv/<q>/<split>_csv/<id>.csv
```

**Output**
- printed AU subset indices/names (console output)

**How to run**
- Edit `data_dir` and `q` inside `au_selection.py` (there is no CLI in the current code).

---

### Step 5: Per-window semantic description generation (LLM)

This step converts each `frame_window_*.csv` into a natural-language description.

**Input**
- `<fau_root>/<q>/<split>_results/<id>/small_window_*/frame_window_*.csv`

**Output**
- `<descriptions_root>/<q>/<split>/<id>.txt`
  - file contains multiple blocks, one per window (separated by blank lines)

**Script**
- `small_windows_description_generation.py` (CLI)

**Command**

```bash
python small_windows_description_generation.py \
  --mode selected \
  --fau-root <base_root> \
  --qx-list q3,q4,q5,q6 \
  --datasets train_results,valid_results,test_results \
  --descriptions-root <descriptions_root> \
  --api-key <YOUR_KEY> \
  --base-url <YOUR_BASE_URL>
```

Important (code caveat):
- The actual model used inside `description_core.call_llm()` is currently hard-coded to `gpt-4o-2024-11-20`.

---

### Step 6: Summary description generation (merge small windows → one summary)

This step merges multiple per-window descriptions into a single summary per video.

**Input**
- `<description_root>/<q>/<split>/<id>.txt` (from Step 5)

**Output**
- `<output_root>/<q>/<split>/<id>.txt` (single merged paragraph)

**Script**
- `summary_descriptions_generation.py` (CLI)

**Command**

```bash
python summary_descriptions_generation.py \
  --mode selected \
  --qx-list q3,q4,q5,q6 \
  --datasets train,valid,test \
  --description-root <descriptions_root> \
  --output-root <processed_descriptions_root> \
  --model gpt-4o-2024-11-20 \
  --api-key <YOUR_KEY> \
  --base-url <YOUR_BASE_URL>
```

---

### Step 7: Build training/eval JSON (full + ablations)

This step builds instruction-tuning style JSON files:
`{instruction, input, output(score)}` for each `(q, split)`.

**Input**
- labels: `annotation_{train,valid,test}.json` (or your own)
- summary descriptions: `<processed_descriptions_root>/<q>/<split>/<id>.txt`
- text transcripts: `<answer_root>/<q>/<split>/text/<id>.txt`

**Output**
- `<output_dir>/<q>_<split>_score_prediction.json`

**Script**
- `generate_data_json.py` (CLI)

**Command (full method = selected AU summaries + text)**

```bash
python generate_data_json.py \
  --mode full \
  --splits train,valid,test \
  --q-list q3,q4,q5,q6 \
  --label-train ./annotation_train.json \
  --label-valid ./annotation_valid.json \
  --label-test  ./annotation_test.json \
  --description-root <processed_descriptions_root> \
  --answer-root <base_root> \
  --output-dir <data_json_dir>
```

**Ablation modes in `generate_data_json.py`**
- `--mode all_au`: use “all AU” descriptions + text
- `--mode only_au`: use selected-AU descriptions only (no text) (legacy alias: `only_AU`)
- `--mode only_text`: use text only (no AU descriptions)

---

### Step 8: Train (LoRA + 4bit) and evaluate

**Training input**
- `<data_json_dir>/<q>_train_score_prediction.json`
- `<data_json_dir>/<q>_valid_score_prediction.json`

**Training output**
- `<save_dir>/best_<q>/` (LoRA adapter)
- `<save_dir>/best_<q>/regression_head.pt`

**Scripts**
- `train.py` (CLI)
- `evaluate.py` (CLI)

**Train**

```bash
python train.py \
  --mode full \
  --q q3 \
  --model-path <HF_BASE_MODEL_PATH> \
  --model-name Llama3 \
  --output-dir <data_json_dir> \
  --save-dir <save_dir> \
  --max-steps 3000 \
  --batch-size 2
```

**Evaluate**

```bash
python evaluate.py \
  --mode full \
  --q q3 \
  --model-name Llama3 \
  --model-path <HF_BASE_MODEL_PATH> \
  --model-dir <save_dir>/best_q3 \
  --test-file <data_json_dir>/q3_test_score_prediction.json \
  --results-path <results_path>
```

---

### Step 9: Statistical analysis / plotting

**Script**
- `statistics.py` 

**Input**
- a folder containing files like: `q3_test_score_prediction_outputs.json` (from `evaluate.py`)

**Output**
- printed metrics + matplotlib figures

**How to run**
- Edit `folder_path` at the bottom of `statistics.py`, then run:

```bash
python statistics.py --folder-path <results_dir> --q-list q3,q4,q5,q6 --save-dir <fig_dir> --no-show
```

---

## 6. Ablation Study

This repo exposes ablation switches as **modes**:

### Training/evaluation modes (`train.py` and `evaluate.py`)

- `--mode full`: selected AU summary + text (default)
- `--mode all_au`: all-AU summary + text
- `--mode only_text`: text only
- `--mode only_au`: selected AU summary only

### Data-generation modes (`generate_data_json.py`)

- `--mode full`
- `--mode all_au`
- `--mode only_text`
- `--mode only_au`

### Mode mapping (canonical + legacy aliases)

| Canonical mode | `generate_data_json.py --mode` | `train.py/evaluate.py --mode` | Legacy aliases still accepted |
|---|---|---|---|
| `full` | `full` | `full` | (none) |
| `all_au` | `all_au` | `all_au` | `ablation_all_AU` |
| `only_text` | `only_text` | `only_text` | `ablation_only_text` |
| `only_au` | `only_au` | `only_au` | `only_AU`, `ablation_only_AU` |

---

## 7. Notes / Known Issues

### 7.1 Windows + OpenFace

- OpenFace `FeatureExtraction.exe` is easiest on Windows, but you must ensure the AU CSV files are saved into:
  - `<base_root>/<q>/<split>_csv/<id>.csv`
- `get_command_tool.py` only prints batched commands; it does not manage output directories by itself.

### 7.2 Frame index alignment (OpenFace vs. video frames)

`KeyFrameExtract.py` matches AU rows by:

- video frame indices from OpenCV (`cv2.CAP_PROP_POS_FRAMES`, stepping every 10 frames)
- exact equality check: `au_data[au_data["frame"] == frame_id]`

If your OpenFace CSV uses a different frame index convention (e.g., 1-based vs 0-based), keyframe windows may contain **missing AU rows** and downstream description generation quality will degrade (or fail silently with empty windows). Always sanity-check a few `(frame_id → AU row)` matches before batch processing.

### 7.3 Hard-coded defaults and path overrides

Several scripts contain Linux absolute default paths like `/root/autodl-tmp/...`. For open-source usage, **always override paths** via CLI:

- `division_tool.py`: `--base-dir`
- `KeyFrameExtract.py`: `--base-root`
- `generate_data_json.py`: `--label-*`, `--description-root`, `--answer-root`, `--output-dir`
- `train.py`: `--output-dir`, `--save-dir`, `--model-path`
- `evaluate.py`: `--model-dir`, `--test-file`, `--results-path`, `--model-path`

### 7.4 LLM model name handling

- `small_windows_description_generation.py` accepts API args, but `description_core.call_llm()` currently uses a fixed model name: `gpt-4o-2024-11-20`.
- `summary_descriptions_generation.py` allows `--model ...` and respects it.

### 7.5 GPU/runtime assumptions

- `train.py` uses 4-bit loading (`BitsAndBytesConfig(load_in_4bit=True)`) and `torch.amp` with `"cuda"`. A CUDA-capable environment is effectively required.
- `dataset_module.py` relies on `tokenizer.apply_chat_template(...)`; your base model/tokenizer should provide a chat template (or you will need to adapt formatting).

---

## 8. Citation

If you use this repository in your research, please cite this work. The citation will be updated upon publication of the paper.
