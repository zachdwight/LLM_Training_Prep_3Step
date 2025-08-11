# Automated Dataset Generation for LLM fine-tuning

This repository provides a three-stage Python pipeline to help you create a high-quality dataset for fine-tuning a Large Language Model (LLM) on the content of a set of PDFs. This approach leverages a local, smaller LLM for initial suggestions and a more powerful LLM (via API or a larger local model) for refining those suggestions into a structured, fine-tuning-ready format.

However, in this example, I've used `MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"` for both tuning steps with a single python script for formatting between them.


This three-step approach leverages:
- ✅ The **efficiency** of a local model for quick, low-cost processing, and
- ✅ The **discipline** of a static script to ensure conversion, and
- ✅ The **capability** of another model for precision and structure (larger if needed).

---

## Step 1

This script is the first stage of the 3-step pipeline. It's responsible for ingesting PDFs, extracting their content, and using a local Large Language Model (LLM) to generate an initial set of Q&A pairs. The output is a raw text file containing the original text chunks and the LLM's suggestions, ready for human review.

How It Works

    Configuration: The script starts by defining key variables like the input and output directories, the strategy for PDF parsing, and the specific LLM to be used. By default, it's configured to use the microsoft/Phi-3-mini-4k-instruct model, which is a lightweight, state-of-the-art open model known for its strong instruction-following capabilities.

    PDF Partitioning: It uses the unstructured.partition_pdf function to break down each PDF document into smaller, structured elements such as paragraphs, titles, and tables. The unstructured library is an open-source ETL (Extract, Transform, Load) solution that effortlessly converts complex, unstructured documents into clean, structured data for language models.

    Text Chunking: The script then processes these elements to create coherent text chunks. It employs a heuristic to combine small, consecutive text snippets into larger, more meaningful blocks, making them more suitable for the LLM to process.

    Local LLM Generation: For each text chunk, the script uses the Hugging Face transformers pipeline to run the selected LLM. It's given a specific prompt to act as an assistant that organizes text into Q&A pairs. This process generates an initial draft of questions and answers.

    Saving for Review: Finally, the script saves the original text chunk and the LLM's generated Q&A suggestions to a .json file for each PDF. This file serves as the input for the next stage, allowing a human to easily review, edit, and curate the generated content.


## Step 2

This script is the crucial human-in-the-loop stage where you review and refine the raw output from Step 1. The goal is to ensure the generated Q&A pairs are high-quality, accurate, and ready to be used for fine-tuning a language model.

How to Curate the Data

    Locate the Output: Navigate to the home/output_json/ directory. Inside, you will find text files (named like [pdf_name]_suggestions.json) that contain the output from the previous script.

    Review and Edit: Open each file and review the Original Text Chunk and the LLM Suggestions. The LLM's suggestions are a first draft and will often contain inaccuracies, formatting errors, or less-than-ideal phrasing.

        Correct Inaccuracies: Check the LLM-generated answers against the original text. If an answer is wrong, edit it to be factually correct.

        Improve Wording: Rework the questions and answers to be clear, concise, and representative of a good training example. The goal is to create data that will teach the LLM to follow instructions and generate accurate responses.

## Step 3

This final script automates the process of generating a clean, high-quality dataset for fine-tuning an LLM. It takes the curated Q&A pairs from Step 2 and uses a local LLM to perform an automated quality check, correcting and filtering the data.

How It Works

    Automated Evaluation: The script loads the curated Q&A pairs and feeds them to the same local LLM used in Step 1. It prompts the model to act as a quality control agent, evaluating each pair for clarity, coherence, and relevance.

    Filtering and Correction:

        "Clear": If an example is labeled as "Clear," it is automatically stripped of any context from the original PDF and added to the final dataset. This ensures the fine-tuning data contains only the essential Q&A format.

        "Unclear" / "Needs Improvement": If an example is flagged, the script attempts to use the LLM to automatically correct the question and then adds the corrected version to the dataset.

        Skipping: Examples that cannot be corrected or are malformed are skipped, preventing low-quality data from contaminating the final dataset.

    Final Output: The script produces two key outputs:

        cleaned_finetune.jsonl: A .jsonl file containing the final, high-quality, and cleaned dataset. This file is ready to be used directly for fine-tuning a language model.

        evaluation.json: A detailed report of the entire process, including the original and corrected messages, and the LLM's evaluation for each entry. This provides a transparent record of the automated curation process.
        Refine Formatting: While the script is designed to parse several formats, maintaining a consistent **Question 1:**... **Answer 1:**... style is best practice for clarity and reliable parsing.

    Prepare for the Next Step: Once you have reviewed and edited all the suggestion files, the curated data is ready for the final processing stage. Do not change the file names or their locations.
