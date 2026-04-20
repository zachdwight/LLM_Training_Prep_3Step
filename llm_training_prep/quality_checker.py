"""Step 3: Evaluate and clean fine-tuning data."""

import json
import re
import logging
from pathlib import Path
from tqdm import tqdm

from .config import Config
from .metrics import MetricsCollector
from .llm_utils import load_llm_pipeline, evaluate_clarity, correct_question

logger = logging.getLogger(__name__)


def parse_evaluation_response(raw_response: str) -> str:
    """Parse evaluation response into a category."""
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


def remove_context(user_message: str) -> str:
    """Remove context lines from user message."""
    lines = user_message.splitlines()
    new_lines = [line for line in lines if not line.strip().lower().startswith("context:")]
    return "\n".join(new_lines).strip()


class QualityChecker:
    """Evaluates and cleans fine-tuning data."""

    def __init__(self, config: Config):
        self.config = config
        self.metrics = MetricsCollector()
        self.llm_pipe = load_llm_pipeline(config.llm.model_id)

    def process_and_filter(
        self,
        jsonl_input: str,
        jsonl_cleaned_output: str,
        json_eval_output: str
    ):
        """Process and filter JSONL data."""
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
                logger.warning(f"Skipping malformed entry: {e}")
                continue

            try:
                eval_response = evaluate_clarity(self.llm_pipe, user_msg, assistant_msg, self.config.llm)
            except Exception as e:
                eval_response = "Error"
                logger.error(f"Evaluation error: {e}")

            eval_result = parse_evaluation_response(eval_response)
            self.metrics.record_evaluation(eval_result)

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
                    question_line = next(
                        (line for line in user_msg.splitlines() if line.lower().startswith("question:")),
                        None
                    )

                    if question_line:
                        raw_question = question_line[len("Question:"):].strip()
                        correction = correct_question(self.llm_pipe, raw_question, self.config.llm)

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
                    else:
                        logger.warning("Could not find question line in user message")
                        result_entry["used_corrected_version"] = False

                except Exception as e:
                    logger.warning(f"Correction failed: {e}")
                    self.metrics.record_correction_error()
                    result_entry["correction_error"] = str(e)
                    result_entry["used_corrected_version"] = False

            else:
                logger.warning(f"Unexpected evaluation label: {eval_result}")
                result_entry["used_corrected_version"] = False

            evaluation_results.append(result_entry)

        final_cleaned = [entry for entry in final_cleaned if entry.get("messages")]

        with open(jsonl_cleaned_output, 'w', encoding='utf-8') as f_out:
            for item in final_cleaned:
                json.dump(item, f_out)
                f_out.write('\n')

        with open(json_eval_output, 'w', encoding='utf-8') as f_eval:
            json.dump(evaluation_results, f_eval, indent=2)

        logger.info(f"Cleaned fine-tuning data written to: {jsonl_cleaned_output}")
        logger.info(f"Evaluation report written to: {json_eval_output}")

        return final_cleaned, evaluation_results
