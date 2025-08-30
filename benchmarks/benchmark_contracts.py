"""Benchmark contracts and interfaces."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field


class BenchmarkTask(str, Enum):
    """Standard benchmark tasks."""
    RETRIEVAL = "retrieval"          # How well does retrieval find relevant docs?
    GENERATION = "generation"        # How good are the generated answers?
    END_TO_END = "end_to_end"       # Complete RAG pipeline evaluation
    RERANKING = "reranking"         # How well does reranking improve results?
    SEMANTIC_SEARCH = "semantic_search"  # Pure semantic similarity


@dataclass
class BenchmarkQuery:
    """A single benchmark query."""
    query_id: str
    query_text: str
    expected_answer: Optional[str] = None
    relevant_doc_ids: Optional[List[str]] = None
    difficulty: Optional[str] = None  # easy, medium, hard
    category: Optional[str] = None    # domain-specific categories
    metadata: Dict[str, Any] = None


@dataclass
class BenchmarkResult:
    """Result of a single query evaluation."""
    query_id: str
    retrieved_docs: List[str]
    generated_answer: Optional[str] = None
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    scores: Dict[str, float] = None  # metric_name -> score


class BenchmarkAdapter(ABC):
    """Abstract adapter for different benchmark datasets."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name."""

    @property
    @abstractmethod
    def tasks(self) -> List[BenchmarkTask]:
        """Supported benchmark tasks."""

    @abstractmethod
    def load_queries(self, split: str = "test") -> List[BenchmarkQuery]:
        """Load benchmark queries."""

    @abstractmethod
    def get_ground_truth(self, query_id: str) -> Dict[str, Any]:
        """Get ground truth for evaluation."""
