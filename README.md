# LLM Training Data Prep: A 3-Step Pipeline

> Turn a folder of PDFs into a clean, fine-tuning-ready JSONL dataset — with a human-in-the-loop review step in the middle.

```
PDFs  ──►  Step 1: extract Q&A suggestions  ──►  *human review*  ──►  Step 2: format  ──►  Step 3: quality-check  ──►  cleaned_finetune.jsonl
```

---

## Setup

```bash
git clone https://github.com/zachdwight/LLM_Training_Prep_3Step.git
cd LLM_Training_Prep_3Step

# Option 1: Install as a package (recommended)
pip install -e .

# Option 2: Just install dependencies
pip install -r requirements.txt
```

> **GPU recommended** — Steps 1 and 3 run a local LLM (`microsoft/Phi-3-mini-4k-instruct` by default).
> If you only have CPU, expect step 1 to be slow. Swap in a smaller model with `--model`.

---

## 🆕 Refactored Package Structure

As of v1.0, this repo is a proper Python package with:
- **Modular design** — reusable classes (`QAGenerator`, `Formatter`, `QualityChecker`)
- **Comprehensive metrics** — JSON reports with quality stats, processing times, and error tracking
- **Configuration management** — centralized config via dataclasses (no hardcoded magic numbers)
- **Proper logging** — structured logging throughout (not just `print()`)
- **Installable** — `pip install -e .` or `pip install .`

**See [REFACTORING.md](REFACTORING.md) for full details on improvements.**

### Quick Start: New Package API

```python
from llm_training_prep.config import Config
from llm_training_prep.qa_generator import QAGenerator

# Create config
config = Config(pdf_dir="./pdfs", output_dir="./output")

# Run Step 1 with metrics
generator = QAGenerator(config)
generator.process_directory()
generator.metrics.print_summary()
generator.metrics.save_report("metrics.json")
```

---

---

## Step 1 — `pdfs_to_ideas_with_llm.py`

Parses each PDF into text chunks, feeds them to a local LLM, and saves the generated Q&A suggestions for human review.

```bash
python pdfs_to_ideas_with_llm.py \
  --pdf-dir    ./pdfs \
  --output-dir ./output \
  --model      microsoft/Phi-3-mini-4k-instruct \
  --strategy   fast
```

| Flag | Default | Description |
|---|---|---|
| `--pdf-dir` | `/home/pdfs/` | Folder of input PDFs |
| `--output-dir` | `/home/output_json/` | Where to write `*_suggestions.json` files |
| `--model` | `microsoft/Phi-3-mini-4k-instruct` | Any HuggingFace causal-LM |
| `--strategy` | `fast` | Unstructured.io strategy: `fast`, `auto`, or `hi_res` |

**Output:** one `[pdf_name]_suggestions.json` per PDF in the output directory.
*(Note: these are structured text files, not JSON — the `.json` extension is intentional for easy identification.)*

---

## Step 2 — Human Review + `format_json.py`

### Human review (do this first)

Open each `*_suggestions.json` in a text editor. For each chunk, check the LLM's Q&A against the `Original Text Chunk` and:

- Correct factual errors
- Improve wording
- Keep the format consistent: `**Question 1:** ... **Answer 1:** ...`

You can also skip review and run Step 2 as-is — Step 3 will catch low-quality pairs automatically.

### Format the files

```bash
python format_json.py \
  --input-dir ./output \
  --output    formatted_finetune.jsonl
```

| Flag | Default | Description |
|---|---|---|
| `--input-dir` | `/home/output_json/` | Folder of `*_suggestions.json` files from Step 1 |
| `--output` | `formatted_finetune.jsonl` | Output JSONL for Step 3 |

**Output:** a JSONL file where each line is a `{"messages": [...]}` object in chat format.

---

## Step 3 — `finalize_tuning_data.py`

Uses the local LLM to evaluate each Q&A pair. Clear pairs pass through as-is; unclear/poor ones get their questions automatically rewritten.

```bash
python finalize_tuning_data.py \
  --input    formatted_finetune.jsonl \
  --output   cleaned_finetune.jsonl \
  --eval-out evaluation.json \
  --model    microsoft/Phi-3-mini-4k-instruct
```

| Flag | Default | Description |
|---|---|---|
| `--input` | `formatted_finetune.jsonl` | JSONL from Step 2 |
| `--output` | `cleaned_finetune.jsonl` | Final fine-tuning dataset |
| `--eval-out` | `evaluation.json` | Evaluation report (per-pair verdict + rewrites) |
| `--model` | `microsoft/Phi-3-mini-4k-instruct` | Local LLM for quality checking |

**Output:**
- `cleaned_finetune.jsonl` — ready to use with any fine-tuning framework (SFTTrainer, Axolotl, etc.)
- `evaluation.json` — audit trail showing per-pair labels and any rewrites

---

## Typical Run (end to end)

```bash
# 1. Generate Q&A suggestions from PDFs
python pdfs_to_ideas_with_llm.py --pdf-dir ./pdfs --output-dir ./output

# 2. (Optional) review ./output/*_suggestions.json in your editor

# 3. Format suggestions into JSONL
python format_json.py --input-dir ./output --output formatted_finetune.jsonl

# 4. Quality-check and finalize
python finalize_tuning_data.py --input formatted_finetune.jsonl
```

---

## Models

The default model is `microsoft/Phi-3-mini-4k-instruct` — it follows instructions reliably, which matters for the structured Q&A format. Other options:

| Model | Notes |
|---|---|
| `microsoft/Phi-3-mini-4k-instruct` | Default — great instruction following |
| `google/gemma-2b-it` | Good quality, slightly larger |
| `prithivMLmods/Llama-Express.1-Tiny` | Very fast, chain-of-thought style output |

---

## Output Format

Each line in the final JSONL is a standard chat-format training example:

```json
{
  "messages": [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "What is X?"},
    {"role": "assistant", "content": "Answer: X is ..."}
  ]
}
```

Compatible with HuggingFace `SFTTrainer`, Axolotl, LlamaFactory, and similar frameworks.

---

## New CLI Commands (Package v1.0+)

After `pip install -e .`, use these convenient CLI commands:

```bash
# Step 1 with metrics output
llm-training-prep-step1 --pdf-dir ./pdfs --output-dir ./output --metrics-output metrics_step1.json

# Step 2
llm-training-prep-step2 --input-dir ./output --output formatted_finetune.jsonl

# Step 3 with quality metrics
llm-training-prep-step3 --input formatted_finetune.jsonl --output cleaned_finetune.jsonl --metrics-output metrics_step3.json
```

---

## Metrics & Reporting

Each step now generates detailed metrics in JSON:

### Step 1 Metrics (e.g., `metrics_step1.json`)
```json
{
  "total_pdfs_processed": 5,
  "total_elements_extracted": 156,
  "total_chunks_created": 42,
  "total_qa_pairs_generated": 350,
  "total_processing_time": 245.3,
  "total_errors": 2
}
```

### Step 3 Quality Metrics (e.g., `metrics_step3.json`)
```json
{
  "quality_metrics": {
    "total_evaluated": 335,
    "clear": 287,
    "unclear": 28,
    "needs_improvement": 20,
    "clear_percentage": 85.7,
    "unclear_percentage": 8.4,
    "needs_improvement_percentage": 5.97
  }
}
```

Use these metrics to:
- Monitor pipeline health (error rates, processing times)
- Assess data quality (% clear vs. needs improvement)
- Debug problematic files
- Track improvements over time

---

## Backward Compatibility

The original scripts (`pdfs_to_ideas_with_llm.py`, `format_json.py`, `finalize_tuning_data.py`) remain unchanged and functional. New users should prefer the refactored package API for better maintainability and metrics.
