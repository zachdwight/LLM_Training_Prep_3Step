"""Shared LLM utilities."""

import torch
from transformers import pipeline
from typing import Dict, List
from .config import LLMConfig


def load_llm_pipeline(model_id: str, task: str = "text-generation"):
    """Load a HuggingFace LLM pipeline."""
    try:
        pipe = pipeline(
            task,
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        return pipe
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_id}: {e}")


def generate_suggestions(
    pipe,
    text_chunk: str,
    prompt_template: str,
    llm_config: LLMConfig
) -> str:
    """Generate suggestions using the LLM."""
    prompt = prompt_template.format(text_chunk=text_chunk)
    messages = [
        {"role": "system", "content": "You are a robot that organizes text into question and answer pairs."},
        {"role": "user", "content": prompt}
    ]

    try:
        outputs = pipe(
            messages,
            max_new_tokens=llm_config.suggestion_max_tokens,
            do_sample=True,
            temperature=llm_config.temperature,
            top_k=llm_config.top_k,
            top_p=llm_config.top_p,
            eos_token_id=pipe.tokenizer.eos_token_id,
            pad_token_id=pipe.tokenizer.eos_token_id
        )

        generated_text = outputs[0]["generated_text"][-1]["content"]

        if generated_text.strip().startswith("<|thinking|>"):
            generated_text = generated_text.split("<|thinking|>", 1)[1].strip()

        return generated_text
    except Exception as e:
        raise RuntimeError(f"Error generating suggestions: {e}")


def evaluate_clarity(pipe, user_input: str, assistant_response: str, llm_config: LLMConfig) -> str:
    """Evaluate clarity of a Q&A pair."""
    prompt = (
        "Analyze the following user-assistant exchange. "
        "Is the question clear? Is the assistant's answer coherent, correct, and relevant? "
        "Respond with one of the following exact words only (no extra explanation): Clear, Unclear, or Needs Improvement.\n\n"
        f"User: {user_input}\n"
        f"Assistant: {assistant_response}"
    )

    try:
        result = pipe(prompt, max_new_tokens=llm_config.clarity_max_tokens)
        return result[0]["generated_text"].strip()
    except Exception as e:
        raise RuntimeError(f"Error evaluating clarity: {e}")


def correct_question(pipe, question: str, llm_config: LLMConfig) -> str:
    """Correct a question using the LLM."""
    prompt = (
        "Rewrite the following question to make it clearer and correct any misspellings, "
        "while preserving its original intent. Do not include any context or extra information.\n\n"
        f"{question}"
    )

    try:
        result = pipe(prompt, max_new_tokens=llm_config.correction_max_tokens)
        return result[0]["generated_text"].strip()
    except Exception as e:
        raise RuntimeError(f"Error correcting question: {e}")
