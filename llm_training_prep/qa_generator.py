"""Step 1: Generate Q&A suggestions from PDFs."""

import os
import logging
import textwrap
from pathlib import Path
from tqdm.auto import tqdm

from .config import Config
from .metrics import MetricsCollector
from .pdf_processor import parse_pdf_to_elements, get_text_chunks_from_elements
from .llm_utils import load_llm_pipeline, generate_suggestions

logger = logging.getLogger(__name__)

SUGGESTION_PROMPT_TEMPLATE = """
Identify and extract the two most important points from the text below and generate two question and answer pairs as if
you are making a FAQ document

text:
{text_chunk}

"""


class QAGenerator:
    """Generates Q&A suggestions from PDFs."""

    def __init__(self, config: Config):
        self.config = config
        self.metrics = MetricsCollector()
        self.llm_pipe = load_llm_pipeline(config.llm.model_id)

    def process_pdf(self, pdf_path: str) -> int:
        """Process a single PDF and return number of suggestions generated."""
        filename = Path(pdf_path).name
        self.metrics.start_file(filename)

        try:
            elements = parse_pdf_to_elements(pdf_path, self.config.pdf)
            self.metrics.record_elements_extracted(len(elements))

            if not elements:
                logger.warning(f"No elements extracted from {pdf_path}")
                self.metrics.record_error(f"No elements extracted from {pdf_path}")
                self.metrics.finish_file()
                return 0

            chunks = get_text_chunks_from_elements(elements, self.config.pdf)
            self.metrics.record_chunks_created(len(chunks))

            output_file = os.path.join(
                self.config.output_dir,
                f"{Path(pdf_path).stem}_suggestions.json"
            )

            if os.path.exists(output_file):
                os.remove(output_file)

            suggestion_count = 0
            for i, chunk in enumerate(tqdm(chunks, desc=f"Processing {filename}")):
                if len(chunk) < self.config.processing.min_chunk_length:
                    continue

                try:
                    suggestions = generate_suggestions(
                        self.llm_pipe,
                        chunk,
                        SUGGESTION_PROMPT_TEMPLATE,
                        self.config.llm
                    )
                    self._save_suggestion(output_file, i + 1, chunk, suggestions)
                    suggestion_count += 1
                except Exception as e:
                    logger.error(f"Error generating suggestion for chunk {i}: {e}")
                    self.metrics.record_error(f"Chunk {i} error: {str(e)}")

            self.metrics.record_qa_pairs(suggestion_count)
            logger.info(f"Generated {suggestion_count} suggestions from {filename}")

        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            self.metrics.record_error(str(e))
        finally:
            self.metrics.finish_file()

        return suggestion_count

    def _save_suggestion(self, output_file: str, chunk_id: int, chunk: str, suggestions: str):
        """Save a suggestion to file."""
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write("-" * 80 + "\n")
            f.write(f"CHUNK ID: {chunk_id}\n")
            f.write("-" * 80 + "\n")
            f.write("Original Text Chunk:\n")
            f.write(textwrap.fill(chunk, width=100))
            f.write("\n\n")
            f.write("LLM Suggestions:\n")
            f.write(textwrap.fill(suggestions, width=100))
            f.write("\n\n")

    def process_directory(self, pdf_dir: str = None) -> int:
        """Process all PDFs in a directory."""
        if pdf_dir is None:
            pdf_dir = self.config.pdf_dir

        os.makedirs(self.config.output_dir, exist_ok=True)

        total_suggestions = 0
        for filename in os.listdir(pdf_dir):
            if not filename.lower().endswith(".pdf"):
                continue

            pdf_path = os.path.join(pdf_dir, filename)
            logger.info(f"Processing {pdf_path}")
            total_suggestions += self.process_pdf(pdf_path)

        logger.info(f"Total suggestions generated: {total_suggestions}")
        return total_suggestions
