import os
import json
import re

SYSTEM_PROMPT = "You are a helpful assistant."

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def extract_chunks(text):
    """Split the text into chunks based on CHUNK ID markers."""
    return re.split(r'-{5,}\s*CHUNK ID:\s*\d+\s*-{5,}', text)

def extract_context_and_llm(chunk_text):
    """Extract the context and LLM Suggestions from one chunk."""
    context_match = re.search(r'Original Text Chunk:\s*(.*?)LLM Suggestions:', chunk_text, re.DOTALL)
    llm_suggestions_match = re.search(r'LLM Suggestions:\s*(.*)', chunk_text, re.DOTALL)

    if not context_match or not llm_suggestions_match:
        return None, None

    context = clean_text(context_match.group(1))
    suggestions = llm_suggestions_match.group(1).strip()

    return context, suggestions

def extract_qa_pairs(suggestions):
    """Extract QA pairs from LLM Suggestions. Supports multiple formats."""
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

    return []  # fallback if nothing matched

def process_file(filepath):
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

    return all_messages

def process_directory(directory):
    all_data = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".json"):  # Still using .json extension for these text files
                filepath = os.path.join(root, filename)
                print(f"📄 Processing: {filename}")
                try:
                    messages = process_file(filepath)
                    print(f"  ➜ Extracted {len(messages)} Q&A pairs.")
                    all_data.extend(messages)
                except Exception as e:
                    print(f"❌ Failed to process {filename}: {e}")
    return all_data

def write_jsonl(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')

if __name__ == "__main__":
    input_directory = "/home/output_json/"
    output_file = "formatted_for_tinyllama.jsonl"

    print(f"📂 Scanning: {input_directory}")
    examples = process_directory(input_directory)
    print(f"✅ Total extracted examples: {len(examples)}")

    write_jsonl(examples, output_file)
    print(f"💾 Output written to: {output_file}")

