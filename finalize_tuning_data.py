import os
import json
import re
# from openai import OpenAI # Remove or comment out this line
from tqdm.auto import tqdm
import google.generativeai as genai # Import the Gemini library

# --- Configuration ---
SUGGESTIONS_INPUT_PATH = "molecular_diagnostics_qa_suggestions.txt"
FINAL_JSONL_OUTPUT_PATH = "molecular_diagnostics_qa.jsonl" # This is the file for fine-tuning





### HEADS UP...there are a couple issue still I'm debugging trying to make this user friendly and stick into docker

# READ ABOVE ^





# --- Larger LLM Configuration (Choose one) ---

# OPTION 1: Google Gemini API (Recommended for reliability and quality)
# Ensure GOOGLE_API_KEY environment variable is set
API_CLIENT_TYPE = "gemini"
API_KEY = os.getenv("GOOGLE_API_KEY") # Changed to GOOGLE_API_KEY
QA_FINALIZATION_MODEL = "gemini-1.5-flash-latest" # Or "gemini-1.5-pro-latest" or other Gemini models

# OPTION 2: Local LLM (e.g., via `ollama` or `vllm` for larger models)
# This requires a local server running your chosen model.
# For example, with Ollama: ollama run llama3 (or mistral, or gemma)
# API_CLIENT_TYPE = "local_ollama"
# QA_FINALIZATION_MODEL = "llama3" # Replace with your locally served model name
# OLLAMA_API_BASE = "http://localhost:11434/v1" # Default Ollama API endpoint

# If using another local server (e.g., vLLM, custom FastAPI),
# you'll need to adapt the client instantiation accordingly.

# Prompt for the *larger* LLM to convert suggestions into structured Q&A
FINALIZATION_PROMPT_TEMPLATE = """
You are an expert in molecular diagnostics and a highly skilled data curator.
Your task is to review the provided "LLM Suggestions" which include a summary and potential questions derived from a textbook chunk.
Your goal is to extract the core informational questions and their precise answers based *only* on the "Original Text Chunk".
You must ensure answers are concise, accurate, and directly supported by the original text.

Strictly follow these rules:
- Generate 2-4 question-answer pairs.
- Each pair must be a direct question and a concise answer.
- The answer must be completely derivable from the "Original Text Chunk". Do NOT invent information.
- The "LLM Suggestions" are *only* ideas; use the "Original Text Chunk" for definitive answers.
- Format the output as a JSON array of objects, where each object has two keys: "instruction" and "response".
- Ensure the JSON is syntactically correct and complete.

Original Text Chunk:
{original_chunk}

LLM Suggestions (for inspiration, but ground answers in Original Text Chunk):
{llm_suggestions}

Output JSON:
"""

# TinyLlama's chat template for the final output (for fine-tuning)
TINYLAMA_CHAT_TEMPLATE = "<|user|>\n{instruction}\n<|assistant|>\n{response}"

# --- Initialize LLM Client ---
def initialize_llm_client(client_type, model_name, api_key=None, api_base=None):
    if client_type == "gemini":
        if api_key is None:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        # For Gemini, we return the GenerativeModel object directly
        return genai.GenerativeModel(model_name)
    elif client_type == "local_ollama":
        if api_base is None:
            raise ValueError("OLLAMA_API_BASE must be set for local_ollama client.")
        # Ollama's API is OpenAI compatible, so we can use the OpenAI client
        from openai import OpenAI # Import OpenAI here if only used for local
        # Return the OpenAI client instance and the model name separately
        return OpenAI(base_url=api_base, api_key="ollama"), model_name # api_key can be anything for local
    else:
        raise ValueError(f"Unsupported API_CLIENT_TYPE: {client_type}")

# Modified initialization to handle Gemini's client structure
if API_CLIENT_TYPE == "gemini":
    llm_client = initialize_llm_client(API_CLIENT_TYPE, QA_FINALIZATION_MODEL, API_KEY)
    QA_FINALIZATION_MODEL_NAME = QA_FINALIZATION_MODEL # Set model name for print statement
else:
    # This part handles OpenAI and local Ollama as before
    llm_client_instance, QA_FINALIZATION_MODEL_NAME = initialize_llm_client(API_CLIENT_TYPE, QA_FINALIZATION_MODEL, API_KEY, OLLAMA_API_BASE if 'OLLAMA_API_BASE' in locals() else None)
    llm_client = llm_client_instance # Assign the client instance

print(f"Initialized LLM client for model: {QA_FINALIZATION_MODEL_NAME}")


# --- Functions ---

def parse_suggestions_file(file_path):
    """
    Parses the molecular_diagnostics_qa_suggestions.txt file
    into a list of dictionaries, each containing 'chunk_id', 'original_chunk', and 'llm_suggestions'.
    """
    chunks_data = []
    current_chunk_data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split the content by the chunk separator
    # Use re.DOTALL to allow . to match newlines
    chunk_separator_pattern = r"-{80}\s*CHUNK ID: (\d+)\s*-{80}"
    parts = re.split(chunk_separator_pattern, content, flags=re.DOTALL)

    # The first part before the first separator is usually empty or preamble, skip it
    for i in range(1, len(parts), 2):
        chunk_id = int(parts[i])
        chunk_content = parts[i+1].strip()

        # Extract Original Text Chunk and LLM Suggestions
        original_text_match = re.search(r"Original Text Chunk:\n(.*?)\n\nLLM Suggestions:", chunk_content, re.DOTALL)
        llm_suggestions_match = re.search(r"LLM Suggestions:\n(.*)", chunk_content, re.DOTALL)

        original_chunk = original_text_match.group(1).strip() if original_text_match else ""
        llm_suggestions = llm_suggestions_match.group(1).strip() if llm_suggestions_match else ""

        if original_chunk: # Only add if we have an original chunk
            chunks_data.append({
                "chunk_id": chunk_id,
                "original_chunk": original_chunk,
                "llm_suggestions": llm_suggestions
            })
    print(f"Parsed {len(chunks_data)} chunks from suggestions file.")
    return chunks_data

def generate_final_qa(original_chunk, llm_suggestions, api_client, model_name, api_client_type):
    """
    Uses the larger LLM to generate final Q&A pairs from the original chunk and suggestions.
    """
    prompt = FINALIZATION_PROMPT_TEMPLATE.format(
        original_chunk=original_chunk,
        llm_suggestions=llm_suggestions
    )
    qa_pairs_raw = ""
    try:
        if api_client_type == "gemini":
            # Gemini's generate_content directly takes the prompt
            response = api_client.generate_content(prompt)
            qa_pairs_raw = response.text
        elif api_client_type == "openai" or api_client_type == "local_ollama":
            messages = [
                {"role": "system", "content": "You are a highly accurate data curator and molecular diagnostics expert."},
                {"role": "user", "content": prompt}
            ]
            response = api_client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"}, # Request JSON response
                temperature=0.0, # Keep temperature low for factual extraction
                max_tokens=500, # Adjust token limit for the expected Q&A output
                seed=42 # For reproducibility if model supports it
            )
            qa_pairs_raw = response.choices[0].message.content
        else:
            raise ValueError(f"Unsupported API_CLIENT_TYPE: {api_client_type}")

        qa_pairs = json.loads(qa_pairs_raw)
        if not isinstance(qa_pairs, list):
            print(f"Warning: LLM returned non-list JSON: {qa_pairs_raw[:100]}...")
            return []
        valid_qa = [p for p in qa_pairs if "instruction" in p and "response" in p and p["instruction"] and p["response"]]
        return valid_qa
    except json.JSONDecodeError as e:
        print(f"JSON decoding error for chunk (might be an LLM formatting issue): {e}")
        print(f"Raw LLM response: {qa_pairs_raw[:500]}...")
        return []
    except Exception as e:
        print(f"Error generating final QA: {e}")
        return []

def append_qa_to_jsonl(qa_data, output_file):
    """Appends Q&A data formatted for TinyLlama to a JSONL file."""
    with open(output_file, 'a', encoding='utf-8') as f:
        for item in qa_data:
            formatted_text = TINYLAMA_CHAT_TEMPLATE.format(
                instruction=item["instruction"],
                response=item["response"]
            )
            json_line = {"text": formatted_text}
            f.write(json.dumps(json_line, ensure_ascii=False) + '\n')


# --- Main Script Execution ---
if __name__ == "__main__":
    if not os.path.exists(SUGGESTIONS_INPUT_PATH):
        print(f"Error: Suggestions file not found at '{SUGGESTIONS_INPUT_PATH}'.")
        print("Please run the previous script (`generate_llm_assisted_qa.py`) first to create it.")
        exit()

    # Remove existing output file if it exists, to start fresh
    if os.path.exists(FINAL_JSONL_OUTPUT_PATH):
        os.remove(FINAL_JSONL_OUTPUT_PATH)
        print(f"Removed existing output file: {FINAL_JSONL_OUTPUT_PATH}")

    # 1. Parse the suggestions file
    chunks_for_finalization = parse_suggestions_file(SUGGESTIONS_INPUT_PATH)

    # 2. Iterate through parsed chunks and generate final Q&A pairs
    total_qa_finalized = 0
    print("\n--- Finalizing Q&A pairs (this may take a while and consume API tokens if using API LLM) ---")
    for i, chunk_data in enumerate(tqdm(chunks_for_finalization, desc="Finalizing Q&A for chunks")):
        chunk_id = chunk_data["chunk_id"] # Make sure chunk_id is available for error messages
        original_chunk = chunk_data["original_chunk"]
        llm_suggestions = chunk_data["llm_suggestions"]

        # You might want to filter out chunks that are too short or seem empty after extraction
        if len(original_chunk.strip()) < 50: # Minimum length for a meaningful Q&A
            # print(f"Skipping very short original chunk (ID: {chunk_id}).")
            continue

        # Pass API_CLIENT_TYPE to the function
        qa_pairs = generate_final_qa(original_chunk, llm_suggestions, llm_client, QA_FINALIZATION_MODEL, API_CLIENT_TYPE)
        if qa_pairs:
            append_qa_to_jsonl(qa_pairs, FINAL_JSONL_OUTPUT_PATH)
            total_qa_finalized += len(qa_pairs)

        # Optional: Add a delay to avoid hitting API rate limits if using a paid API
        # import time
        # time.sleep(0.1) (google gemini 1.5 should be set at 4.0 due to 15 RPM

    print(f"\nFinished finalizing Q&A pairs. Total pairs generated: {total_qa_finalized}")
    print(f"Final fine-tuning dataset saved to: {FINAL_JSONL_OUTPUT_PATH}")
    print("\n--- Next Steps ---")
    print("1. Your `molecular_diagnostics_qa.jsonl` file is now ready.")
    print("2. Use this JSONL file for fine-tuning TinyLlama/TinyLlama-1.1B-Chat-v1.0 (as shown in the previous example).")
    print("   Remember to load the dataset using: `dataset = load_dataset('json', data_files='molecular_diagnostics_qa.jsonl')`")
    print("   And set `formatting_func=lambda example: example['text']` in your SFTTrainer.")
    print("3. Always perform human evaluation on a sample of the fine-tuned model's output to ensure quality.")
