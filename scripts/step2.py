#!/usr/bin/env python
"""Step 2: Parse suggestion files into JSONL fine-tuning format."""

import argparse
import logging

from llm_training_prep.formatter import Formatter


def main():
    """Main entry point for Step 2."""
    parser = argparse.ArgumentParser(
        description="Step 2: Parse suggestion files into JSONL fine-tuning format."
    )
    parser.add_argument("--input-dir", default="/home/output_json/",
                        help="Directory of *_suggestions.json files from Step 1")
    parser.add_argument("--output", default="formatted_finetune.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting Step 2: Formatting to JSONL")

    formatter = Formatter()
    examples = formatter.process_directory(args.input_dir)

    formatter.write_jsonl(examples, args.output)

    logger.info(f"Step 2 complete. Total examples: {len(examples)}")
    logger.info(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
