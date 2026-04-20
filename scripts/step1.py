#!/usr/bin/env python
"""Step 1: Extract Q&A pairs from PDFs using a local LLM."""

import argparse
import logging
import os

from llm_training_prep.config import Config
from llm_training_prep.qa_generator import QAGenerator


def main():
    """Main entry point for Step 1."""
    parser = argparse.ArgumentParser(
        description="Step 1: Extract Q&A pairs from PDFs using a local LLM."
    )
    parser.add_argument("--pdf-dir", default="/home/pdfs/", help="Directory containing PDF files")
    parser.add_argument("--output-dir", default="/home/output_json/", help="Directory for suggestions")
    parser.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct", help="HuggingFace model ID")
    parser.add_argument("--strategy", default="fast", choices=["auto", "fast", "hi_res"],
                        help="PDF parsing strategy")
    parser.add_argument("--metrics-output", default="metrics_step1.json", help="Metrics output file")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Create config
    config = Config(
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
    )
    config.llm.model_id = args.model
    config.pdf.strategy = args.strategy

    os.makedirs(args.output_dir, exist_ok=True)

    # Run generator
    logger.info("Starting Step 1: Q&A Generation")
    generator = QAGenerator(config)

    total_suggestions = generator.process_directory(args.pdf_dir)

    # Save metrics
    generator.metrics.save_report(args.metrics_output)
    generator.metrics.print_summary()

    logger.info(f"Step 1 complete. Total suggestions: {total_suggestions}")
    logger.info(f"Metrics saved to: {args.metrics_output}")


if __name__ == "__main__":
    main()
