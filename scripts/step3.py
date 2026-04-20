#!/usr/bin/env python
"""Step 3: Evaluate and clean fine-tuning data using a local LLM."""

import argparse
import logging

from llm_training_prep.config import Config
from llm_training_prep.quality_checker import QualityChecker


def main():
    """Main entry point for Step 3."""
    parser = argparse.ArgumentParser(
        description="Step 3: Evaluate and clean fine-tuning data using a local LLM."
    )
    parser.add_argument("--input", default="formatted_finetune.jsonl",
                        help="JSONL file from Step 2")
    parser.add_argument("--output", default="cleaned_finetune.jsonl",
                        help="Output cleaned JSONL")
    parser.add_argument("--eval-out", default="evaluation.json",
                        help="Evaluation report JSON")
    parser.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--metrics-output", default="metrics_step3.json",
                        help="Metrics output file")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting Step 3: Quality Checking")

    # Create config
    config = Config()
    config.llm.model_id = args.model

    # Run quality checker
    checker = QualityChecker(config)
    final_cleaned, eval_results = checker.process_and_filter(
        args.input,
        args.output,
        args.eval_out
    )

    # Save metrics
    checker.metrics.save_report(args.metrics_output)
    checker.metrics.print_summary()

    logger.info(f"Step 3 complete. Final cleaned examples: {len(final_cleaned)}")
    logger.info(f"Metrics saved to: {args.metrics_output}")


if __name__ == "__main__":
    main()
