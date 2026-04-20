"""Configuration for LLM training data prep pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PDFConfig:
    """PDF parsing configuration."""
    max_characters: int = 1000
    new_after_n_chars: int = 800
    overlap: int = 150
    chunk_combine_threshold: int = 500
    chunk_refinement_threshold: int = 1000
    strategy: str = "fast"  # "auto", "fast", "hi_res"


@dataclass
class LLMConfig:
    """LLM pipeline configuration."""
    model_id: str = "microsoft/Phi-3-mini-4k-instruct"
    suggestion_max_tokens: int = 300
    clarity_max_tokens: int = 150
    correction_max_tokens: int = 100
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95


@dataclass
class ProcessingConfig:
    """Processing pipeline configuration."""
    min_chunk_length: int = 100
    debug_first_n_elements: int = 20


@dataclass
class Config:
    """Main configuration for the entire pipeline."""
    pdf: PDFConfig = None
    llm: LLMConfig = None
    processing: ProcessingConfig = None

    pdf_dir: str = "/home/pdfs/"
    output_dir: str = "/home/output_json/"

    def __post_init__(self):
        if self.pdf is None:
            self.pdf = PDFConfig()
        if self.llm is None:
            self.llm = LLMConfig()
        if self.processing is None:
            self.processing = ProcessingConfig()
