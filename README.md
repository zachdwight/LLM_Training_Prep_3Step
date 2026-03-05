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

pip install -r requirements.txt
```

> **GPU recommended** — Steps 1 and 3 run a local LLM (`microsoft/Phi-3-mini-4k-instruct` by default).
> If you only have CPU, expect step 1 to be slow. Swap in a smaller model with `--model`.

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
