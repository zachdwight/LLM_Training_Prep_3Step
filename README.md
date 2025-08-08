# Automated Dataset Generation for LLM fine-tuning

This repository provides a two-stage Python pipeline to help you create a high-quality dataset for fine-tuning a Large Language Model (LLM) on the content of a PDF textbook, our example being molecular diagnostics. This approach leverages a local, smaller LLM for initial suggestions and a more powerful LLM (via API or a larger local model) for refining those suggestions into a structured, fine-tuning-ready format.   

Please note, Google Gemini & OpenAI were used to organize / rewrite some of my code and documention for better understanding (hopefully no mismatches in file names, etc).  2nd Note - different APIs or models may cause JSON issues.  I've just updated both scripts with some improvements (Jun 19 2025).   Thanks!


This two-step approach leverages:
- ✅ The **efficiency** of a local model for quick, low-cost processing, and
- ✅ The **capability** of a larger model for precision and structure.

---

## Table of Contents

1.  [Introduction](#introduction)
2.  [Features](#features)
3.  [Prerequisites](#prerequisites)
4.  [Setup](#setup)
    * [API Keys](#api-keys)
    * [Installation](#installation)
    * [PDF Textbook](#pdf-textbook)
5.  [Usage](#usage)
    * [Step 1: Generate LLM-Assisted Q&A Suggestions](#step-1-generate-llm-assisted-qa-suggestions)
    * [Step 2: Finalize Q&A Data with a Stronger LLM](#step-2-finalize-qa-data-with-a-stronger-llm)
6.  [Output Files](#output-files)
7.  [Next Steps: Fine-tuning Your LLM](#next-steps-fine-tuning-your-llm)
8.  [Important Notes & Considerations](#important-notes--considerations)
9.  [License](#license)

---

## 1. Introduction (Two-Step Pipeline)

### Step 1: Local Model for Initial Suggestions

- **Objective**:  
  Utilize a smaller, local LLM to generate preliminary suggestions from a PDF textbook.

- **Process**:
  - Extract text from the PDF.
  - Input extracted content into the local LLM.
  - Generate initial suggestions or summaries.

- **Output**:  
  A set of preliminary suggestions ready for refinement.

---

### Step 2: Large Model for Refinement

- **Objective**:  
  Employ a more powerful LLM (via API or a larger local model) to refine the initial suggestions into a structured, fine-tuning-ready format.

- **Process**:
  - Input preliminary suggestions into the large LLM.
  - Refine and structure the suggestions.

- **Output**:  
  A high-quality dataset suitable for fine-tuning an LLM.

---


## 2. Features

* **Robust PDF Parsing:** Utilizes `Unstructured.io` for intelligent extraction of text, titles, and tables from complex PDF layouts.
* **Semantic Chunking:** Organizes extracted content into coherent, context-rich chunks.
* **Local LLM Assistance:** Employs your specified local LLM (`prithivMLmods/Llama-Express.1-Tiny` in the example but also try `microsoft/Phi-3-mini-4k-instruct` or `google/gemma-2b-it`) to provide summaries and question ideas for each chunk.  Both Google and Microsoft options are less chatty (instruct vs chat style).
* **Automated Q&A Finalization:** Leverages a powerful LLM (e.g., Google Gemini or OpenAI, or a strong self-hosted model) to convert suggestions into structured Q&A pairs.
* **Fine-tuning Ready Output:** Generates a `.jsonl` file formatted for direct use with Hugging Face's `SFTTrainer` (e.g., for `TinyLlama/TinyLlama-1.1B-Chat-v1.0`).

---

## 3. Prerequisites

* **Python 3.8+**
* A **GPU** is highly recommended for running local LLMs and for eventual fine-tuning.
* **Sufficient RAM** for running local LLMs and processing PDF data.
* **Access to a powerful LLM API** (e.g., OpenAI) OR a powerful self-hosted LLM (e.g., Llama 3 8B Instruct, Mixtral) running locally via `ollama`, `vLLM`, etc., for the final Q&A generation step.

---

## 4. Setup

### API Keys

---
* **Google Gemini API Key:** To use Google Gemini for Q&A generation (or other AI tasks), you'll need an API key from Google AI Studio.

    1.  Go to [Google AI Studio](https://aistudio.google.com/app/apikey) and sign in with your Google account.
    2.  Follow the prompts to **create a new API key**.
    3.  Once generated, set it as an **environment variable**. The recommended variable name is `GEMINI_API_KEY`.

    ```bash
    # For Linux/macOS
    export GEMINI_API_KEY='your_api_key_here'
    # For Windows (Command Prompt)
    set GEMINI_API_KEY=your_api_key_here
    # For Windows (PowerShell)
    $env:GEMINI_API_KEY='your_api_key_here'
    ```
    (It's a good idea to add this to your shell's profile file, like `~/.bashrc` or `~/.zshrc`, so it's always set.)

**Important Security Note:** Never hardcode your API key directly into your application's code, especially for client-side applications. Always use environment variables or a secure secret management service.


### Installation

Clone this repository and install the required Python packages:

```bash
git clone https://github.com/zachdwight/LLM_TrainingPrep_Local2Large.git
cd LLM_TrainingPrep_Local2Large

# Install core Python dependencies
pip install unstructured["pdf"] transformers torch tqdm openai reportlab # reportlab is for creating a dummy PDF if needed

# Additional system-level dependencies for Unstructured.io (especially for 'hi_res' strategy and OCR)
# For Debian/Ubuntu:
sudo apt-get install poppler-utils tesseract-ocr
# For macOS (using Homebrew):
brew install poppler tesseract

# If you plan to use Ollama locally for the finalization step:
# Follow the Ollama installation instructions: [https://ollama.com/](https://ollama.com/)
# Then pull your desired model, for example:
# ollama pull llama3
# Make sure `ollama serve` is running in the background when you use the script.
```

### PDF Textbook

Place your molecular diagnostics textbook PDF file in the root directory of this repository and name it molecular_diagnostics_textbook.pdf and be respectful of others copyright.

## 5. Usage

This pipeline consists of two sequential Python scripts.
Step 1: Generate LLM-Assisted Q&amp;A Suggestions

Run this script (generate_llm_assisted_qa.py) to parse your PDF, chunk its content, and use your local Llama-Express.1-Tiny model to generate summaries and potential questions for each chunk. However, I have seen other options perform better such as Google's Gemma or Microsoft's PHI-3 smaller models.  These models tend to follow directions better so it is good to compare your local LLM options.  The output is a human-readable text file that you'll use for refinement in the next step.

```
python local_LLM_data_prep.py
```

What it does:

    - Reads molecular_diagnostics_textbook.pdf.
    - Uses Unstructured.io for advanced PDF text extraction and semantic chunking.
    - Loads your local LLM (prithivMLmods/Llama-Express.1-Tiny).
    - For each text chunk, it prompts your local LLM to generate a summary and a list of suggested questions.
    - Saves these suggestions, along with the original chunk, to molecular_diagnostics_qa_suggestions.txt.

Step 2: Finalize Q&amp;A Data with a Stronger LLM

Next, run this script (finalize_tuning_data.py). It takes the suggestions from Step 1 and uses a more powerful LLM (e.g., OpenAI's GPT-4o, or a locally served Llama 3) to convert them into a structured JSON Lines (.jsonl) dataset. This is the dataset ready for fine-tuning.

Before running, you'll need to configure the LLM client directly in finalize_tuning_data.py:

```
# Open `finalize_qa_data.py` and choose your client type by commenting/uncommenting:

# OPTION 1: OpenAI API (Recommended for reliability and quality)
API_CLIENT_TYPE = "openai"
API_KEY = os.getenv("OPENAI_API_KEY") # Ensure this env var is set
QA_FINALIZATION_MODEL = "gpt-4o" # You can also try "gpt-4-turbo", "gpt-3.5-turbo" for lower cost

# OPTION 2: Local Ollama (if you have an Ollama server running with your desired model, e.g., llama3)
# API_CLIENT_TYPE = "local_ollama"
# QA_FINALIZATION_MODEL = "llama3" # Replace with the exact name of your locally served model
# OLLAMA_API_BASE = "http://localhost:11434/v1" # Default Ollama API endpoint, adjust if yours is different
```
Once configured, run the script:

```
python finalize_tuning_data.py
```
What it does:

    - Reads molecular_diagnostics_qa_suggestions.txt.
    - Connects to the specified stronger LLM (OpenAI API or local Ollama).
    - For each entry from the suggestions file, it prompts the powerful LLM with the original text chunk and the local LLM's suggestions.
    - It instructs the LLM to generate 2-4 highly accurate, concise, and grounded Q&amp;A pairs in a strict JSON format.
    - Saves the resulting Q&amp;A pairs into molecular_diagnostics_qa.jsonl, which is formatted for SFTTrainer.

## 6. Output Files

    - molecular_diagnostics_qa_suggestions.txt: This is an intermediate output from Step 1. It's a human-readable file containing original text chunks and suggestions from your local LLM. While it's mainly for the second script, you can review it for debugging or understanding the initial LLM output.
    - molecular_diagnostics_qa.jsonl: This is the final output from Step 2. It's a meticulously prepared dataset, where each line is a JSON object with a text key. This text key contains a formatted instruction-response pair (<|user|>\nYour question here\n<|assistant|>\nYour answer here). This file is ready for LLM fine-tuning.

## 7. Next Steps: Fine-tuning Your LLM

With molecular_diagnostics_qa.jsonl in hand, you're ready to fine-tune TinyLlama/TinyLlama-1.1B-Chat-v1.0 (or any other compatible instruction-tuned model) using Hugging Face's transformers and trl libraries.

When setting up your fine-tuning script, remember these key points:

    Load your dataset using Hugging Face's datasets library:
   
```
from datasets import load_dataset
dataset = load_dataset('json', data_files='molecular_diagnostics_qa.jsonl')
```
Ensure your formatting_func for SFTTrainer directly returns the text field from your dataset, as it's already pre-formatted:
Python
```
    formatting_func=lambda example: example['text']
```
## 8. Important Notes & Considerations

    - Cost: Using powerful LLM APIs (like OpenAI's GPT-4o) for finalize_qa_data.py will incur costs based on token usage. It's wise to monitor your API dashboard to keep track of expenses.
    - Quality Control (Human Review): While this pipeline automates a significant portion of the data generation, manual review of a sample of molecular_diagnostics_qa.jsonl is highly recommended, especially for critical applications. Even powerful LLMs can occasionally hallucinate, misinterpret context, or produce less-than-ideal answers.
    - LLM Capabilities: The quality of the generated Q&amp;A (especially in Step 2) is directly dependent on the capabilities of the LLM you choose for finalization. Generally, stronger models will yield better results.
    - PDF Complexity: This pipeline works best with well-structured PDFs. If your textbook has a very complex layout, is a scanned PDF (without high-quality OCR), or has fragmented text, you might find that Unstructured.io struggles, potentially requiring additional preprocessing or manual intervention.
    - Chunking Strategy: The Unstructured.io parameters like chunking_strategy and max_characters in generate_llm_assisted_qa.py are quite important. You might need to experiment with these values to find the optimal chunk size that provides enough context for Q&amp;A without overwhelming the LLMs.
    - Prompt Engineering: The prompts used in both scripts (LOCAL_LLM_SUGGESTION_PROMPT_TEMPLATE and FINALIZATION_PROMPT_TEMPLATE) are crucial. You might need to fine-tune them based on the specific content of your textbook and the desired style and depth of your Q&amp;A.

## 9. License

This project is open-source and available under the MIT License.
