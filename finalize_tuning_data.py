import json
from transformers import pipeline
from tqdm import tqdm
import re

# === Model Setup ===
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
print(f"🔄 Loading local LLM: {MODEL_ID}...")

local_llm_pipe = pipeline(
    "text-generation",
    model=MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    max_new_tokens=150,
    do_sample=False,
    return_full_text=False
)

# === Prompts ===

def build_clarity_prompt(user_input, assistant_response):
    return (
        "Analyze the following user-assistant exchange. "
        "Is the question clear? Is the assistant's answer coherent, correct, and relevant? "
        "Respond with one of the following exact words only (no extra explanation): Clear, Unclear, or Needs Improvement.\n\n"
        f"User: {user_input}\n"
        f"Assistant: {assistant_response}"
    )

def build_correction_prompt(question):
    return (
        "Rewrite the following question to make it clearer and correct any misspellings, "
        "while preserving its original intent. Do not include any context or extra information.\n\n"
        f"{question}"
    )

def remove_context(user_message):
    """
    Remove lines starting with 'Context:' from the user message.
    """
    lines = user_message.splitlines()
    new_lines = [line for line in lines if not line.strip().lower().startswith("context:")]
    return "\n".join(new_lines).strip()

def parse_evaluation_response(raw_response):
    """
    Normalize and classify model's output into: clear, unclear, or needs improvement.
    """
    if not raw_response:
        return "unknown"

    text = raw_response.lower()

    if "needs improvement" in text:
        return "needs improvement"
    elif "unclear" in text:
        return "unclear"
    elif "clear" in text:
        return "clear"
    else:
        return "unknown"

# === Main Process ===

def process_and_filter(jsonl_input, jsonl_cleaned_output, json_eval_output):
    with open(jsonl_input, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    final_cleaned = []
    evaluation_results = []

    for line in tqdm(lines, desc="Evaluating examples"):
        item = json.loads(line)
        messages = item.get("messages", [])
        try:
            user_msg = next(m["content"] for m in messages if m["role"] == "user")
            assistant_msg = next(m["content"] for m in messages if m["role"] == "assistant")
        except Exception as e:
            print(f"⚠️ Skipping malformed entry: {e}")
            continue

        clarity_prompt = build_clarity_prompt(user_msg, assistant_msg)

        try:
            eval_response = local_llm_pipe(clarity_prompt)[0]["generated_text"].strip()
        except Exception as e:
            eval_response = "Error"
            print(f"❌ Evaluation error: {e}")

        print(f"\n📝 Raw evaluation response: {eval_response!r}")
        eval_result = parse_evaluation_response(eval_response)
        print(f"⚙️ Parsed evaluation label: {eval_result!r}")

        result_entry = {
            "original_messages": messages,
            "evaluation": eval_response,
            "parsed_evaluation": eval_result
        }

        if eval_result == "clear":
            cleaned_user_msg = remove_context(user_msg)

            new_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": cleaned_user_msg},
                {"role": "assistant", "content": assistant_msg}
            ]

            final_cleaned.append({"messages": new_messages})
            result_entry["used_corrected_version"] = False

        elif eval_result in {"unclear", "needs improvement"}:
            try:
                question_line = next(line for line in user_msg.splitlines() if line.lower().startswith("question:"))
                raw_question = question_line[len("Question:"):].strip()

                correction_prompt = build_correction_prompt(raw_question)
                correction = local_llm_pipe(correction_prompt)[0]["generated_text"].strip()

                cleaned_user_msg = remove_context(user_msg)

                new_user_msg = re.sub(
                    re.escape(raw_question),
                    correction,
                    cleaned_user_msg,
                    flags=re.IGNORECASE
                )

                new_messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": new_user_msg},
                    {"role": "assistant", "content": assistant_msg}
                ]

                final_cleaned.append({"messages": new_messages})
                result_entry["corrected_question"] = correction
                result_entry["used_corrected_version"] = True

            except Exception as e:
                print(f"⚠️ Correction failed: {e}")
                result_entry["correction_error"] = str(e)
                result_entry["used_corrected_version"] = False

        else:
            print(f"⚠️ Unexpected evaluation label: {eval_result}. Treating as Unclear and skipping entry.")
            result_entry["used_corrected_version"] = False

        evaluation_results.append(result_entry)

    final_cleaned = [entry for entry in final_cleaned if entry.get("messages")]

    with open(jsonl_cleaned_output, 'w', encoding='utf-8') as f_out:
        for item in final_cleaned:
            json.dump(item, f_out)
            f_out.write('\n')

    with open(json_eval_output, 'w', encoding='utf-8') as f_eval:
        json.dump(evaluation_results, f_eval, indent=2)

    print(f"\n✅ Cleaned fine-tuning data written to: {jsonl_cleaned_output}")
    print(f"📝 Evaluation report written to: {json_eval_output}")

if __name__ == "__main__":
    input_jsonl = "formatted_for_tinyllama.jsonl"
    output_jsonl = "cleaned_finetune.jsonl"
    eval_json = "evaluation.json"

    process_and_filter(input_jsonl, output_jsonl, eval_json)



