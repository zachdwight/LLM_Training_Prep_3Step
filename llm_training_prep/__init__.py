"""LLM Training Data Prep - Convert PDFs to fine-tuning datasets."""

__version__ = "1.0.0"

from .metrics import MetricsCollector
from .config import Config

__all__ = ["MetricsCollector", "Config"]
