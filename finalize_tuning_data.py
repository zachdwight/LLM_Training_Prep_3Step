import os
import json
import re
import time
# from openai import OpenAI # Remove or comment out this line
from tqdm.auto import tqdm
import google.generativeai as genai # Import the Gemini library

# --- Configuration ---
SUGGESTIONS_INPUT_PATH = "molecular_diagnostics_qa_suggestions.txt"
FINAL_JSONL_OUTPUT_PATH = "molecular_diagnostics_qa.jsonl" # This is the file for fine-tuning

# --- Larger LLM Configuration (Choose one) ---
os.environ['MY_API_KEY'] = 'your_api_key_here' #



# OPTION 1: Google Gemini API (Recommended for reliability and quality)
# Ensure GOOGLE_API_KEY environment variable is set
API_CLIENT_TYPE = "gemini"
API_KEY = os.environ.get('MY_API_KEY') # Changed to GOOGLE_API_KEY
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
You are an expert in biostatistics and a highly skilled data curator.
Your task is to review the provided "LLM Suggestions" which include a summary and potential questions derived from a textbook chunk.
Your goal is to extract the core informational questions and their precise answers based *only* on the "Original Text Chunk".
You must ensure answers are concise, accurate, and directly supported by the original text.

Strictly follow these rules:
- Generate 2 question-answer pairs that a biostats student would need to know.
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
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    chunk_separator_pattern = r"-{80}\s*CHUNK ID: (\d+)\s*-{80}"
    parts = re.split(chunk_separator_pattern, content, flags=re.DOTALL)

    print(f"\n[DEBUG_PARSE] Total parts after splitting by chunk separator: {len(parts)}")
    if len(parts) <= 1:
        print("[DEBUG_PARSE] No chunks found or separator not matched. Content head:")
        print(content[:500]) # Print first 500 chars if no chunks
        return []

    # Iterate through parts, skipping the first empty one
    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts): # Ensure there's content after the ID
            print(f"[DEBUG_PARSE] Incomplete chunk block detected at index {i}.")
            continue

        chunk_id = int(parts[i])
        chunk_content = parts[i+1].strip()

        print(f"\n[DEBUG_PARSE] Processing CHUNK ID: {chunk_id}")
        # print(f"[DEBUG_PARSE] Raw chunk_content (first 300 chars):\n{chunk_content[:300]}...") # Uncomment for very detailed debug


        # Regex for Original Text Chunk
        original_text_match = re.search(r"Original Text Chunk:\n(.*?)\n\nLLM Suggestions:", chunk_content, re.DOTALL)
        original_chunk = original_text_match.group(1).strip() if original_text_match else ""

        # Regex for LLM Suggestions (VERY ROBUST VERSION)
        # It now looks for "LLM Suggestions:", then optionally any whitespace,
        # then optionally <|thinking|> (escaped), then optionally any more whitespace,
        # and then captures everything else.
        llm_suggestions_match = re.search(r"LLM Suggestions:\s*(<\|thinking\|>)?\s*(.*)", chunk_content, re.DOTALL)
        llm_suggestions = llm_suggestions_match.group(2).strip() if llm_suggestions_match else ""


        print(f"[DEBUG_PARSE] Extracted original_chunk len: {len(original_chunk)}")
        print(f"[DEBUG_PARSE] Extracted llm_suggestions len: {len(llm_suggestions)}")

        if len(original_chunk) > 50 and len(llm_suggestions) > 50: # Ensure both are substantial
            chunks_data.append({
                "chunk_id": chunk_id,
                "original_chunk": original_chunk,
                "llm_suggestions": llm_suggestions
            })
            print(f"[DEBUG_PARSE] CHUNK ID {chunk_id} added successfully.")
        else:
            print(f"[DEBUG_PARSE_WARNING] Skipping CHUNK ID: {chunk_id} due to short original_chunk ({len(original_chunk)}) or llm_suggestions ({len(llm_suggestions)}).")
            if not original_text_match:
                print(f"[DEBUG_PARSE_WARNING] 'Original Text Chunk:' pattern not found for ID {chunk_id}.")
            if not llm_suggestions_match:
                print(f"[DEBUG_PARSE_WARNING] 'LLM Suggestions:' pattern not found for ID {chunk_id}.")


    print(f"\nParsed {len(chunks_data)} valid chunks from suggestions file.")
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
    json_string_to_parse = "" # Initialize for error reporting

    try:
        # Debug: Check the API Key status (do NOT print the actual key!)
        # Use os.getenv for environment variable check if applicable
        # This check might need slight adjustment based on how API_KEY is truly sourced
        if api_client_type == "gemini":
            if os.getenv("GOOGLE_API_KEY") is None and API_KEY is None:
                print("[DEBUG_GEMINI_ERROR] GOOGLE_API_KEY environment variable is not set, and API_KEY in script is None.")
                return []
            elif API_KEY == "YOUR_GEMINI_API_KEY_HERE" or API_KEY == "AIzaSyA3oYU1HJa_-WSWWdAvxx6vk0585HDY3zI": # Check for your specific placeholder key
                 print("[DEBUG_GEMINI_WARNING] Using placeholder API_KEY in script. Ensure it's a valid, active key.")
            elif not API_KEY or API_KEY == "":
                print("[DEBUG_GEMINI_ERROR] API_KEY is set but empty or invalid in script configuration.")
                return []


        print(f"[DEBUG_GEMINI] Sending prompt to LLM (first 500 chars):\n{prompt[:500]}...")

        if api_client_type == "gemini":
            try:
                response = api_client.generate_content(prompt)
                # Check if response has text attribute before accessing
                if hasattr(response, 'text'):
                    qa_pairs_raw = response.text
                else:
                    print(f"[DEBUG_GEMINI_ERROR] Gemini response object has no 'text' attribute. Full response: {response}")
                    return []
            except Exception as api_err:
                print(f"[DEBUG_GEMINI_ERROR] Gemini API call failed: {api_err}")
                print(f"  Possible causes: Invalid API Key, Rate Limit Exceeded, Network Issue, Model Not Found.")
                return []
        elif api_client_type == "openai" or api_client_type == "local_ollama":
            messages = [
                {"role": "system", "content": "You are a highly accurate data curator and molecular diagnostics expert."},
                {"role": "user", "content": prompt}
            ]
            try:
                response = api_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    response_format={"type": "json_object"}, # Request JSON response
                    temperature=0.0, # Keep temperature low for factual extraction
                    max_tokens=3000, # Adjust token limit for the expected Q&A output
                    seed=42 # For reproducibility if model supports it
                )
                qa_pairs_raw = response.choices[0].message.content
            except Exception as api_err:
                print(f"[DEBUG_LLM_ERROR] Local/OpenAI LLM API call failed: {api_err}")
                return []
        else:
            raise ValueError(f"Unsupported API_CLIENT_TYPE: {api_client_type}")

        print(f"[DEBUG_GEMINI] Raw LLM response received (first 500 chars):\n{qa_pairs_raw[:500]}...")

        if not qa_pairs_raw.strip():
            print("[DEBUG_GEMINI_WARNING] LLM returned an empty or whitespace-only response.")
            return []

        # --- NEW CODE TO EXTRACT JSON FROM MARKDOWN BLOCK ---
        # Regex to find content inside ```json ... ```
        json_match = re.search(r"```json\s*(.*?)\s*```", qa_pairs_raw, re.DOTALL)
        if json_match:
            json_string_to_parse = json_match.group(1).strip()
            print("[DEBUG_GEMINI] Successfully extracted JSON string from Markdown block.")
        else:
            # If it's not in a markdown block, assume it's pure JSON (for robustness)
            json_string_to_parse = qa_pairs_raw.strip()
            print("[DEBUG_GEMINI_WARNING] No JSON Markdown block found, attempting to parse raw response directly.")
        # --- END NEW CODE ---

        # Use the extracted string for json.loads()
        qa_pairs = json.loads(json_string_to_parse)
        print(f"[DEBUG_GEMINI] Successfully parsed JSON. Found {len(qa_pairs)} items.")

        if not isinstance(qa_pairs, list):
            print(f"[DEBUG_GEMINI_WARNING] LLM returned non-list JSON (expected a list of objects). Raw: {json_string_to_parse[:100]}...")
            return []

        # Filter out pairs with empty instruction or response after stripping whitespace
        valid_qa = [p for p in qa_pairs if "instruction" in p and "response" in p and p["instruction"].strip() and p["response"].strip()]
        print(f"[DEBUG_GEMINI] Number of valid Q&A pairs after filtering (non-empty instruction/response): {len(valid_qa)}")

        return valid_qa
    except json.JSONDecodeError as e:
        print(f"\n[ERROR] JSON decoding error for chunk (LLM formatting issue): {e}")
        print(f"[ERROR] Raw LLM response that caused JSON error:\n{qa_pairs_raw[:500]}...")
        print(f"[ERROR] String attempting to parse: {json_string_to_parse[:500]}") # Show the string that failed to parse
        return []
    except Exception as e:
        print(f"\n[ERROR] General error in generate_final_qa: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for unexpected errors
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
    print(f"Initialized LLM client for model: {QA_FINALIZATION_MODEL_NAME}")

    print(f"\n[DEBUG_MAIN] Total chunks prepared for finalization loop: {len(chunks_for_finalization)}")
    if not chunks_for_finalization:
        print("[CRITICAL] No chunks to process after parsing. Exiting.")
        exit() # Exit immediately if no chunks were parsed
    else:
        # Print a snippet of the first parsed chunk to confirm content
        print(f"[DEBUG_MAIN] First chunk's original_chunk (first 100 chars): {chunks_for_finalization[0]['original_chunk'][:100]}...")
        print(f"[DEBUG_MAIN] First chunk's llm_suggestions (first 100 chars): {chunks_for_finalization[0]['llm_suggestions'][:100]}...")


    # 2. Iterate through parsed chunks and generate final Q&A pairs
    total_qa_finalized = 0
    print("\n--- Finalizing Q&A pairs (this may take a while and consume API tokens if using API LLM) ---")
    for i, chunk_data in enumerate(tqdm(chunks_for_finalization, desc="Finalizing Q&A for chunks")):
        chunk_id = chunk_data["chunk_id"]
        original_chunk = chunk_data["original_chunk"]
        llm_suggestions = chunk_data["llm_suggestions"]

        if len(original_chunk.strip()) < 50: # Minimum length for a meaningful Q&A
            print(f"[DEBUG_MAIN_SKIP] Skipping very short original chunk (ID: {chunk_id}, length: {len(original_chunk.strip())}).")
            continue

        print(f"[DEBUG_MAIN] Attempting to generate QA for Chunk ID: {chunk_id}")
        qa_pairs = generate_final_qa(original_chunk, llm_suggestions, llm_client, QA_FINALIZATION_MODEL, API_CLIENT_TYPE)

        if qa_pairs:
            print(f"[DEBUG_MAIN] Successfully generated {len(qa_pairs)} QA pairs for Chunk ID: {chunk_id}. Appending to JSONL.")
            append_qa_to_jsonl(qa_pairs, FINAL_JSONL_OUTPUT_PATH)
            total_qa_finalized += len(qa_pairs)
        else:
            print(f"[DEBUG_MAIN_FAIL] No QA pairs generated for Chunk ID: {chunk_id}.")

        # Crucial for Gemini 1.5 Flash free tier (15 RPM)
        # 60 seconds / 15 requests = 4 seconds per request minimum
        time.sleep(4.0)

    print(f"\nFinished finalizing Q&A pairs. Total pairs generated: {total_qa_finalized}")
    print(f"Final fine-tuning dataset saved to: {FINAL_JSONL_OUTPUT_PATH}")

    print("\n--- Next Steps ---")
    print("1. Your `molecular_diagnostics_qa.jsonl` file is now ready.")
    print("2. Use this JSONL file for fine-tuning TinyLlama/TinyLlama-1.1B-Chat-v1.0 (as shown in the previous example).")
    print("   Remember to load the dataset using: `dataset = load_dataset('json', data_files='molecular_diagnostics_qa.jsonl')`")
    print("   And set `formatting_func=lambda example: example['text']` in your SFTTrainer.")
    print("3. Always perform human evaluation on a sample of the fine-tuned model's output to ensure quality.")
