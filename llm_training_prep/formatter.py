"""Step 2: Format suggestions into JSONL fine-tuning format."""

import os
import json
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a helpful assistant."


def clean_text(text: str) -> str:
    """Normalize whitespace in text."""
    return re.sub(r'\s+', ' ', text).strip()


def extract_chunks(text: str) -> list:
    """Split text into chunks based on CHUNK ID markers."""
    return re.split(r'-{5,}\s*CHUNK ID:\s*\d+\s*-{5,}', text)


def extract_context_and_llm(chunk_text: str) -> tuple:
    """Extract context and LLM suggestions from a chunk."""
    context_match = re.search(r'Original Text Chunk:\s*(.*?)LLM Suggestions:', chunk_text, re.DOTALL)
    llm_suggestions_match = re.search(r'LLM Suggestions:\s*(.*)', chunk_text, re.DOTALL)

    if not context_match or not llm_suggestions_match:
        return None, None

    context = clean_text(context_match.group(1))
    suggestions = llm_suggestions_match.group(1).strip()

    return context, suggestions


def extract_qa_pairs(suggestions: str) -> list:
    """Extract Q&A pairs from LLM suggestions."""
    qa_pairs = []

    # Try format 1: **Question 1:** ... **Answer 1:** ...
    pattern_qa = re.findall(
        r"\*\*Question\s*\d*\*\*:\s*(.*?)\s*\*\*Answer\s*\d*\*\*:\s*(.*?)(?=\*\*Question\s*\d*\*\*:|$)",
        suggestions,
        re.DOTALL,
    )
    if pattern_qa:
        for q, a in pattern_qa:
            qa_pairs.append((clean_text(q), clean_text(a)))
        return qa_pairs

    # Try format 2: **Q1:** ... A1: ...
    pattern_faq = re.findall(
        r"\*\*Q\d+\s*:\*\*\s*(.*?)\s*A\d+\s*:\s*(.*?)(?=\*\*Q\d+\s*:\*\*|$)",
        suggestions,
        re.DOTALL,
    )
    if pattern_faq:
        for q, a in pattern_faq:
            qa_pairs.append((clean_text(q), clean_text(a)))
        return qa_pairs

    # Try plain FAQ without bold markers
    pattern_plain_faq = re.findall(
        r"Q\d+:\s*(.*?)\s*A\d+:\s*(.*?)(?=Q\d+:|$)",
        suggestions,
        re.DOTALL,
    )
    if pattern_plain_faq:
        for q, a in pattern_plain_faq:
            qa_pairs.append((clean_text(q), clean_text(a)))
        return qa_pairs

    return []


class Formatter:
    """Formats suggestions into JSONL fine-tuning format."""

    def __init__(self):
        self.total_files = 0
        self.total_qa_pairs = 0

    def process_file(self, filepath: str) -> list:
        """Process a single suggestion file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        chunks = extract_chunks(raw_text)
        all_messages = []

        for chunk in chunks:
            context, suggestions = extract_context_and_llm(chunk)
            if not context or not suggestions:
                continue

            qa_pairs = extract_qa_pairs(suggestions)
            for question, answer in qa_pairs:
                user = f"Question: {question}\nContext: {context}"
                assistant = f"Answer: {answer}"

                all_messages.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant}
                    ]
                })
                self.total_qa_pairs += 1

        return all_messages

    def process_directory(self, directory: str) -> list:
        """Process all suggestion files in a directory."""
        all_data = []

        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.endswith(".json"):
                    filepath = os.path.join(root, filename)
                    logger.info(f"Processing: {filename}")

                    try:
                        messages = self.process_file(filepath)
                        logger.info(f"  Extracted {len(messages)} Q&A pairs from {filename}")
                        all_data.extend(messages)
                        self.total_files += 1
                    except Exception as e:
                        logger.error(f"Failed to process {filename}: {e}")

        return all_data

    def write_jsonl(self, data: list, output_file: str):
        """Write formatted data to JSONL file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f)
                f.write('\n')

        logger.info(f"Wrote {len(data)} examples to {output_file}")
