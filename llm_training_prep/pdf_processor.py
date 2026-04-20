"""PDF processing utilities."""

import logging
from typing import List
from unstructured.partition.pdf import partition_pdf
from .config import PDFConfig

logger = logging.getLogger(__name__)


def parse_pdf_to_elements(pdf_path: str, config: PDFConfig):
    """Extract structured elements from PDF."""
    try:
        elements = partition_pdf(
            filename=pdf_path,
            strategy=config.strategy,
            infer_table_structure=True,
            extract_images_as_bytes=False,
            max_characters=config.max_characters,
            new_after_n_chars=config.new_after_n_chars,
            overlap=config.overlap,
        )
        logger.info(f"Extracted {len(elements)} elements from {pdf_path}")
        return elements
    except Exception as e:
        logger.error(f"Error parsing PDF {pdf_path}: {e}")
        raise


def get_text_chunks_from_elements(elements, config: PDFConfig) -> List[str]:
    """Convert elements into coherent text chunks."""
    text_chunks = []
    current_chunk = ""

    for element in elements:
        if hasattr(element, "text") and element.text:
            text_to_add = element.text.strip()
            if text_to_add:
                if len(current_chunk) < config.chunk_combine_threshold and current_chunk:
                    current_chunk += "\n" + text_to_add
                else:
                    if current_chunk:
                        text_chunks.append(current_chunk.strip())
                    current_chunk = text_to_add
        elif hasattr(element, "category") and element.category == "Table" and element.text:
            table_text = f"Table content:\n{element.text.strip()}"
            if len(table_text) > 50:
                if current_chunk:
                    text_chunks.append(current_chunk.strip())
                    current_chunk = ""
                text_chunks.append(table_text)

    if current_chunk:
        text_chunks.append(current_chunk.strip())

    refined_chunks = []
    temp_chunk = ""
    for chunk in text_chunks:
        if len(temp_chunk) + len(chunk) < config.chunk_refinement_threshold and temp_chunk:
            temp_chunk += "\n" + chunk
        else:
            if temp_chunk:
                refined_chunks.append(temp_chunk)
            temp_chunk = chunk
    if temp_chunk:
        refined_chunks.append(temp_chunk)

    logger.info(f"Created {len(refined_chunks)} text chunks")
    return refined_chunks
