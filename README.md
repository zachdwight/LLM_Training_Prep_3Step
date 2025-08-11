# Automated Dataset Generation for LLM fine-tuning

This repository provides a three-stage Python pipeline to help you create a high-quality dataset for fine-tuning a Large Language Model (LLM) on the content of a set of PDFs. This approach leverages a local, smaller LLM for initial suggestions and a more powerful LLM (via API or a larger local model) for refining those suggestions into a structured, fine-tuning-ready format.

However, in this example, I've used `MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"` for both tuning steps with a single python script for formatting between them.


This three-step approach leverages:
- ✅ The **efficiency** of a local model for quick, low-cost processing, and
- ✅ The **discipline** of a static script to ensure conversion, and
- ✅ The **capability** of another model for precision and structure (larger if needed).

---
