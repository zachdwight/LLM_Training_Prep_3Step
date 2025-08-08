import os
import json
from unstructured.partition.pdf import partition_pdf
print(partition_pdf)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm.auto import tqdm
import textwrap
os.environ["OCR_AGENT"] = "unstructured_pytesseract"

# --- Configuration ---
PDF_PATH = "/home/biostats.pdf" # Replace with your PDF file
SUGGESTIONS_OUTPUT_PATH = "biostats_suggestions.txt" # Output file for human review

# Some ideas for a local LLM
#MODEL_ID = "prithivMLmods/Llama-Express.1-Tiny"  #very chatty and chain of thought so could be good for certain applications
#MODEL_ID = "google/gemma-2b-it" #performs well
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct" #follows instructions super well so I prefer in this case.  

# Parameters for Unstructured.io PDF parsing
UNSTRUCTURED_STRATEGY = "auto" # "auto", "fast", "hi_res" - hi_res can be slower but more accurate
UNSTRUCTURED_STRATEGY = "fast"  # or "auto"
# Prompt for the local LLM to generate suggestions
# Emphasize summary or question ideas, not structured Q&A JSON.
LOCAL_LLM_SUGGESTION_PROMPT_TEMPLATE = """
Context:
{text_chunk}

Based on the above context from a biostatistics textbook, provide a 1 sentence summary of two key points, and then list 2 potential questions that could be answered using only this text.
Strictly follow this output format:

Summary:
- [Your summary point 1]
- [Your summary point 2]

Questions:
- [Question 1]
- [Question 2]
"""

# --- Functions ---

def parse_pdf_to_elements(pdf_path, strategy=UNSTRUCTURED_STRATEGY):
    """
    Parses a PDF using Unstructured.io to extract structured elements.
    Returns a list of dicts, where each dict is an element (e.g., paragraph, title, table).
    """
    print(f"Parsing PDF: {pdf_path} using strategy: {strategy}...")
    try:
        elements = partition_pdf(
            filename=pdf_path,
            strategy=strategy,
            infer_table_structure=True,
            extract_images_as_bytes=False,
            chunking_strategy="by_title",
            max_characters=2000,
            new_after_n_chars=1500,
            overlap=150,
        )
        print(f"Extracted {len(elements)} elements.")
        return elements
    except Exception as e:
        print(f"Error parsing PDF with Unstructured.io: {e}")
        print("Ensure 'unstructured' and its dependencies are installed correctly.")
        print("For 'hi_res' strategy, you might need extra dependencies like 'poppler-utils' and 'tesseract'.")
        return []

def get_text_chunks_from_elements(elements):
    """
    Extracts relevant text chunks from Unstructured elements, focusing on narrative text.
    Combines small narrative text elements to form larger, more coherent chunks.
    """
    text_chunks = []
    current_chunk = ""
    for element in elements:
        if hasattr(element, "text") and element.text:
            text_to_add = element.text.strip()
            if text_to_add:
                # Simple heuristic to combine small text bits
                if len(current_chunk) < 500 and current_chunk:
                    current_chunk += "\n" + text_to_add
                else:
                    if current_chunk:
                        text_chunks.append(current_chunk.strip())
                    current_chunk = text_to_add
        elif element.category == "Table" and element.text:
            table_text = f"Table content:\n{element.text.strip()}"
            if len(table_text) > 50:
                 if current_chunk:
                    text_chunks.append(current_chunk.strip())
                    current_chunk = ""
                 text_chunks.append(table_text)

    if current_chunk:
        text_chunks.append(current_chunk.strip())

    # Further refinement: combine very small chunks that are consecutive
    refined_chunks = []
    temp_chunk = ""
    for chunk in text_chunks:
        if len(temp_chunk) + len(chunk) < 1000 and temp_chunk:
            temp_chunk += "\n" + chunk
        else:
            if temp_chunk:
                refined_chunks.append(temp_chunk)
            temp_chunk = chunk
    if temp_chunk:
        refined_chunks.append(temp_chunk)

    return refined_chunks

def get_llm_suggestions(text_chunk, llm_pipeline):
    """
    Uses the local LLM to generate summary and question suggestions from a text chunk.
    """
    prompt = LOCAL_LLM_SUGGESTION_PROMPT_TEMPLATE.format(text_chunk=text_chunk)
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides concise summaries and question ideas."},
        {"role": "user", "content": prompt}
    ]
    try:
        outputs = llm_pipeline(
            messages,
            max_new_tokens=300, # Adjust token limit for suggestions
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            # Ensure these match your model's tokenizer
            eos_token_id=llm_pipeline.tokenizer.eos_token_id,
            pad_token_id=llm_pipeline.tokenizer.eos_token_id
        )
        # Extract the generated text from the last message in the conversation
        generated_text = outputs[0]["generated_text"][-1]["content"]
        if generated_text.strip().startswith("<|thinking|>"):
            generated_text = generated_text.split("<|thinking|>", 1)[1].strip()
        return generated_text
    except Exception as e:
        print(f"Error getting LLM suggestions for chunk: {e}")
        return f"Error: Could not generate suggestions ({e})"

def save_suggestions_for_review(chunk_id, original_chunk, llm_suggestions, output_file):
    """Saves the original chunk and LLM suggestions to a text file for human review."""
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("-" * 80 + "\n")
        f.write(f"CHUNK ID: {chunk_id}\n")
        f.write("-" * 80 + "\n")
        f.write("Original Text Chunk:\n")
        f.write(textwrap.fill(original_chunk, width=100))
        f.write("\n\n")
        f.write("LLM Suggestions:\n")
        f.write(textwrap.fill(llm_suggestions, width=100))
        f.write("\n\n")


# --- Main Script Execution ---
if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF file not found at '{PDF_PATH}'. Please place your molecular diagnostics textbook PDF there.")
        # Create a dummy PDF for quick testing if needed
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            dummy_text = """
            Chapter 1: Introduction to Molecular Diagnostics.
            Molecular diagnostics is a field that involves the detection and analysis of DNA, RNA, and proteins to diagnose diseases,
            assess prognosis, and guide therapeutic decisions. It has revolutionized medicine by enabling precise and early detection.

            Common techniques include Polymerase Chain Reaction (PCR), Next-Generation Sequencing (NGS), and Microarrays.
            PCR is used to amplify specific DNA sequences. It involves denaturation, annealing, and extension steps.
            Real-time PCR (qPCR) allows for quantification of DNA/RNA in real-time.

            Chapter 2: DNA Extraction Methods.
            DNA extraction is the process of isolating DNA from biological samples.
            Common methods include organic extraction, solid-phase extraction, and magnetic bead-based extraction.
            Each method has advantages and disadvantages depending on the sample type and downstream application.
            """
            c = canvas.Canvas(PDF_PATH, pagesize=letter)
            textobject = c.beginText()
            textobject.setTextOrigin(50, 750)
            for line in dummy_text.strip().split('\n'):
                textobject.textLine(line.strip())
            c.drawText(textobject)
            c.save()
            print(f"Created a dummy PDF at '{PDF_PATH}' for demonstration.")
        except ImportError:
            print("Install 'reportlab' (pip install reportlab) to create dummy PDF, or provide your own.")
            exit()

    # Initialize the local LLM pipeline
    print(f"Loading local LLM: {MODEL_ID}...")
    try:
        # Use your exact pipeline setup from the provided example
        local_llm_pipe = pipeline(
            "text-generation",
            model=MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto", # Use "auto" for flexibility, or "cuda" if you're sure
        )
        print("Local LLM pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading local LLM pipeline: {e}")
        print("Please ensure your model path is correct and dependencies are installed.")
        exit()

    # 1. Parse PDF into elements
    elements = parse_pdf_to_elements(PDF_PATH)
    if not elements:
        print("No elements extracted from PDF. Exiting.")
        exit()
    for i, el in enumerate(elements[:20]):  # show first 20 for debugging
        print(f"Element {i}: category={el.category}, text={el.text[:60]}")
    # 2. Convert elements into coherent text chunks for Q&A generation
    text_chunks_for_qa = get_text_chunks_from_elements(elements)
    print(f"Prepared {len(text_chunks_for_qa)} text chunks for Q&A generation.")

    # Remove existing output file if it exists, to start fresh
    if os.path.exists(SUGGESTIONS_OUTPUT_PATH):
        os.remove(SUGGESTIONS_OUTPUT_PATH)
        print(f"Removed existing output file: {SUGGESTIONS_OUTPUT_PATH}")

    # 3. Iterate through chunks and get LLM suggestions
    print("\n--- Generating LLM suggestions for human review ---")
    for i, chunk in enumerate(tqdm(text_chunks_for_qa, desc="Processing chunks for suggestions")):
        if len(chunk) < 100: # Skip very small chunks
            continue

        llm_suggestions = get_llm_suggestions(chunk, local_llm_pipe)
        save_suggestions_for_review(i + 1, chunk, llm_suggestions, SUGGESTIONS_OUTPUT_PATH)
