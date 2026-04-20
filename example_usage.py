#!/usr/bin/env python
"""Example usage of the refactored llm_training_prep package."""

import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_step1_programmatic():
    """Example: Run Step 1 programmatically with custom config."""
    from llm_training_prep.config import Config
    from llm_training_prep.qa_generator import QAGenerator

    logger.info("=== Example: Step 1 - Q&A Generation ===")

    # Create config with custom settings
    config = Config(
        pdf_dir="/path/to/pdfs",
        output_dir="/path/to/output"
    )

    # Customize LLM settings
    config.llm.model_id = "microsoft/Phi-3-mini-4k-instruct"
    config.llm.suggestion_max_tokens = 300

    # Customize PDF parsing
    config.pdf.max_characters = 1000
    config.pdf.strategy = "fast"

    # Initialize generator
    generator = QAGenerator(config)

    # Process directory
    total_suggestions = generator.process_directory()

    # Print and save metrics
    generator.metrics.print_summary()
    generator.metrics.save_report("metrics_step1.json")

    logger.info(f"Generated {total_suggestions} suggestions")
    logger.info("Metrics saved to metrics_step1.json")


def example_step2_programmatic():
    """Example: Run Step 2 programmatically."""
    from llm_training_prep.formatter import Formatter

    logger.info("=== Example: Step 2 - Format to JSONL ===")

    formatter = Formatter()

    # Process directory
    examples = formatter.process_directory("/path/to/output")

    # Write to JSONL
    formatter.write_jsonl(examples, "formatted_finetune.jsonl")

    logger.info(f"Formatted {len(examples)} Q&A pairs")


def example_step3_programmatic():
    """Example: Run Step 3 programmatically with metrics."""
    from llm_training_prep.config import Config
    from llm_training_prep.quality_checker import QualityChecker

    logger.info("=== Example: Step 3 - Quality Checking ===")

    config = Config()
    config.llm.model_id = "microsoft/Phi-3-mini-4k-instruct"

    checker = QualityChecker(config)

    # Process and filter
    final_cleaned, eval_results = checker.process_and_filter(
        jsonl_input="formatted_finetune.jsonl",
        jsonl_cleaned_output="cleaned_finetune.jsonl",
        json_eval_output="evaluation.json"
    )

    # Print and save metrics
    checker.metrics.print_summary()
    checker.metrics.save_report("metrics_step3.json")

    logger.info(f"Cleaned {len(final_cleaned)} Q&A pairs")


def example_accessing_metrics():
    """Example: Accessing metrics programmatically."""
    from llm_training_prep.metrics import MetricsCollector

    logger.info("=== Example: Accessing Metrics ===")

    collector = MetricsCollector()

    # Simulate some processing
    collector.start_file("example.pdf")
    collector.record_elements_extracted(50)
    collector.record_chunks_created(10)
    collector.record_qa_pairs(20)
    collector.finish_file()

    # Simulate quality evaluation
    collector.record_evaluation("clear")
    collector.record_evaluation("clear")
    collector.record_evaluation("unclear")
    collector.record_evaluation("needs improvement")

    # Access metrics
    print(f"Total PDFs: {collector.metrics.total_pdfs_processed}")
    print(f"Total chunks: {collector.metrics.total_chunks_created}")

    # Quality breakdown
    qm = collector.metrics.quality_metrics
    print(f"Quality: {qm.clear}/{qm.total_evaluated} clear ({qm.clear_percentage:.1f}%)")

    # Save report
    collector.save_report("metrics_example.json")


def example_custom_config():
    """Example: Creating a custom config."""
    from llm_training_prep.config import Config, LLMConfig, PDFConfig

    logger.info("=== Example: Custom Config ===")

    # Option 1: Modify default config
    config = Config()
    config.pdf.max_characters = 2000
    config.llm.temperature = 0.5

    # Option 2: Create custom LLM config
    llm_config = LLMConfig(
        model_id="llama2",
        suggestion_max_tokens=500,
        temperature=0.3
    )

    # Option 3: Create custom PDF config
    pdf_config = PDFConfig(
        max_characters=1500,
        strategy="hi_res"
    )

    logger.info("Config examples created successfully")


if __name__ == "__main__":
    logger.info("LLM Training Prep - Usage Examples")
    logger.info("")
    logger.info("Available examples:")
    logger.info("  1. example_step1_programmatic() - Run Step 1 with custom config")
    logger.info("  2. example_step2_programmatic() - Run Step 2 with metrics")
    logger.info("  3. example_step3_programmatic() - Run Step 3 with quality metrics")
    logger.info("  4. example_accessing_metrics() - Access metrics programmatically")
    logger.info("  5. example_custom_config() - Create custom configurations")
    logger.info("")
    logger.info("To run an example, call the function in Python:")
    logger.info("  python -c 'from example_usage import example_step1_programmatic; example_step1_programmatic()'")
