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
