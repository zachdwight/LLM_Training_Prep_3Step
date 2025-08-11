# LLM Training Data Prep: A 3-Step Pipeline

This repository provides a 3-step pipeline for preparing high-quality, fine-tuning data for Large Language Models (LLMs). The process automates the extraction of content from PDFs, generates initial Q&A pairs using a local LLM, allows for a crucial human-in-the-loop curation step, and then performs a final automated quality check to create a clean, ready-to-use dataset.

***

### 1. ⚙️ Step 1: `pdfs_to_ideas_with_llm.py`

This script is the first stage of the pipeline. It takes a collection of PDFs, extracts their text content, and uses a local LLM to generate an initial set of Q&A pairs. The output is a raw text file for each PDF, containing both the original content chunks and the LLM's suggestions.

#### How It Works

* **PDF Parsing**: It uses the `unstructured` library to partition each PDF into structured elements like paragraphs and tables.
* **Text Chunking**: The elements are then combined into coherent, context-rich text chunks suitable for LLM processing.
* **LLM Generation**: A local LLM (pre-configured to use **`microsoft/Phi-3-mini-4k-instruct`**) is prompted to generate two question-and-answer pairs for each chunk.
* **Output**: The suggestions are saved to a `.json` file (named `[pdf_name]_suggestions.json`) in the `/home/output_json/` directory, ready for human review.

***

### 2. ✍️ Step 2: `format_json.py`

This is the **human-in-the-loop stage** where you manually review and refine the raw output from Step 1. The goal is to ensure the Q&A pairs are accurate, clear, and high-quality before they are used for training.

#### How to Curate the Data

1.  **Review**: Open the `[pdf_name]_suggestions.json` files in the `/home/output_json/` directory.
2.  **Edit**:
    * **Correct** any factual inaccuracies in the LLM-generated answers by checking them against the `Original Text Chunk`.
    * **Improve** the wording of both questions and answers to make them concise and effective for training.
    * **Maintain Consistency**: A consistent format like `**Question 1:**... **Answer 1:**...` is recommended for reliable parsing in the next step.

***

### 3. ✅ Step 3: `finalize_tuning_data.py`

This final script automates the process of generating a clean, high-quality dataset. It takes the curated Q&A pairs from Step 2 and uses a local LLM to perform an automated quality check, correcting and filtering the data.

#### How It Works

* **Automated Evaluation**: The script uses a local LLM as a quality control agent, evaluating each Q&A pair for clarity and coherence.
* **Filtering**:
    * **"Clear" examples** are stripped of their context and added to the final dataset.
    * **"Unclear" or "Needs Improvement" examples** have their questions automatically rewritten by the LLM and are then added to the final dataset.
* **Final Output**: The script produces two files:
    * `cleaned_finetune.jsonl`: The final, clean dataset ready for fine-tuning.
    * `evaluation.json`: A detailed report of the automated quality control process.
