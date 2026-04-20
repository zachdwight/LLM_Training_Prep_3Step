"""Metrics collection and reporting for the pipeline."""

import json
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


@dataclass
class FileMetrics:
    """Metrics for a single file."""
    filename: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    elements_extracted: int = 0
    chunks_created: int = 0
    qa_pairs_generated: int = 0
    qa_pairs_kept: int = 0
    errors: List[str] = field(default_factory=list)
    processing_time: float = 0.0

    def finish(self):
        self.end_time = time.time()
        self.processing_time = self.end_time - self.start_time


@dataclass
class QualityMetrics:
    """Quality evaluation metrics."""
    total_evaluated: int = 0
    clear: int = 0
    unclear: int = 0
    needs_improvement: int = 0
    correction_errors: int = 0

    @property
    def clear_percentage(self) -> float:
        if self.total_evaluated == 0:
            return 0.0
        return (self.clear / self.total_evaluated) * 100

    @property
    def unclear_percentage(self) -> float:
        if self.total_evaluated == 0:
            return 0.0
        return (self.unclear / self.total_evaluated) * 100

    @property
    def needs_improvement_percentage(self) -> float:
        if self.total_evaluated == 0:
            return 0.0
        return (self.needs_improvement / self.total_evaluated) * 100


@dataclass
class PipelineMetrics:
    """Overall pipeline metrics."""
    total_pdfs_processed: int = 0
    total_elements_extracted: int = 0
    total_chunks_created: int = 0
    total_qa_pairs_generated: int = 0
    total_qa_pairs_kept: int = 0
    total_processing_time: float = 0.0
    total_errors: int = 0
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    quality_metrics: QualityMetrics = field(default_factory=QualityMetrics)
    file_metrics: Dict[str, FileMetrics] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data['quality_metrics'] = asdict(self.quality_metrics)
        data['file_metrics'] = {k: asdict(v) for k, v in self.file_metrics.items()}
        return data


class MetricsCollector:
    """Collects and reports metrics throughout the pipeline."""

    def __init__(self):
        self.metrics = PipelineMetrics()
        self._current_file: Optional[FileMetrics] = None

    def start_file(self, filename: str):
        """Start tracking a file."""
        self._current_file = FileMetrics(filename=filename)
        self.metrics.file_metrics[filename] = self._current_file

    def finish_file(self):
        """Finish tracking current file."""
        if self._current_file:
            self._current_file.finish()
            self.metrics.total_pdfs_processed += 1
            self.metrics.total_elements_extracted += self._current_file.elements_extracted
            self.metrics.total_chunks_created += self._current_file.chunks_created
            self.metrics.total_qa_pairs_generated += self._current_file.qa_pairs_generated
            self.metrics.total_qa_pairs_kept += self._current_file.qa_pairs_kept
            self.metrics.total_processing_time += self._current_file.processing_time
            self.metrics.total_errors += len(self._current_file.errors)

    def record_elements_extracted(self, count: int):
        """Record number of elements extracted from PDF."""
        if self._current_file:
            self._current_file.elements_extracted = count

    def record_chunks_created(self, count: int):
        """Record number of text chunks created."""
        if self._current_file:
            self._current_file.chunks_created = count

    def record_qa_pairs(self, generated: int, kept: int = None):
        """Record Q&A pairs generated."""
        if self._current_file:
            self._current_file.qa_pairs_generated = generated
            if kept is not None:
                self._current_file.qa_pairs_kept = kept

    def record_error(self, error: str):
        """Record an error."""
        if self._current_file:
            self._current_file.errors.append(error)
        self.metrics.total_errors += 1

    def record_evaluation(self, result: str):
        """Record quality evaluation result."""
        self.metrics.quality_metrics.total_evaluated += 1
        if result == "clear":
            self.metrics.quality_metrics.clear += 1
        elif result == "unclear":
            self.metrics.quality_metrics.unclear += 1
        elif result == "needs improvement":
            self.metrics.quality_metrics.needs_improvement += 1

    def record_correction_error(self):
        """Record a correction error."""
        self.metrics.quality_metrics.correction_errors += 1

    def save_report(self, output_path: str):
        """Save metrics report to JSON."""
        self.metrics.end_time = datetime.now().isoformat()

        report = self.metrics.to_dict()

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

    def print_summary(self):
        """Print a summary of metrics."""
        print("\n" + "=" * 80)
        print("PIPELINE METRICS SUMMARY")
        print("=" * 80)

        print(f"\nProcessing Summary:")
        print(f"  PDFs processed: {self.metrics.total_pdfs_processed}")
        print(f"  Total elements extracted: {self.metrics.total_elements_extracted}")
        print(f"  Total chunks created: {self.metrics.total_chunks_created}")
        print(f"  Total Q&A pairs generated: {self.metrics.total_qa_pairs_generated}")
        print(f"  Total Q&A pairs kept: {self.metrics.total_qa_pairs_kept}")
        print(f"  Total processing time: {self.metrics.total_processing_time:.2f}s")
        print(f"  Total errors: {self.metrics.total_errors}")

        if self.metrics.quality_metrics.total_evaluated > 0:
            qm = self.metrics.quality_metrics
            print(f"\nQuality Evaluation Results:")
            print(f"  Total evaluated: {qm.total_evaluated}")
            print(f"  Clear: {qm.clear} ({qm.clear_percentage:.1f}%)")
            print(f"  Unclear: {qm.unclear} ({qm.unclear_percentage:.1f}%)")
            print(f"  Needs improvement: {qm.needs_improvement} ({qm.needs_improvement_percentage:.1f}%)")
            print(f"  Correction errors: {qm.correction_errors}")

        print("\n" + "=" * 80)
