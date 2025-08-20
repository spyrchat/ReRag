"""
Contracts and schemas for the ingestion pipeline.
Defines the social contract between raw data and the pipeline.
"""
import hashlib
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Iterable, Union
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, validator
from langchain_core.documents import Document


class DatasetSplit(str, Enum):
    """Standardized dataset splits."""
    TRAIN = "train"
    VALIDATION = "val" 
    TEST = "test"
    ALL = "all"


class BaseRow(BaseModel):
    """Base schema for dataset-specific rows. All adapters must extend this."""
    external_id: str = Field(..., description="Original dataset identifier")
    
    class Config:
        extra = "allow"  # Allow dataset-specific fields


class ChunkMeta(BaseModel):
    """Dataset-agnostic metadata for processed chunks."""
    # Identity
    doc_id: str = Field(..., description="Deterministic document ID")
    chunk_id: str = Field(..., description="Deterministic chunk ID")
    
    # Content provenance
    doc_sha256: str = Field(..., description="SHA256 of normalized document content")
    text: str = Field(..., description="Chunk text content")
    
    # Source tracking
    source: str = Field(..., description="Dataset/source name")
    dataset_version: str = Field(..., description="Dataset version")
    external_id: str = Field(..., description="Original dataset identifier")
    uri: Optional[str] = Field(None, description="Source URI/path")
    
    # Processing metadata
    chunk_index: int = Field(..., description="0-based chunk index within document")
    num_chunks: int = Field(..., description="Total chunks in document")
    
    # Content metadata
    token_count: Optional[int] = Field(None, description="Estimated token count")
    char_count: int = Field(..., description="Character count")
    
    # Dataset metadata
    split: DatasetSplit = Field(..., description="Dataset split")
    labels: Dict[str, Any] = Field(default_factory=dict, description="Dataset labels/annotations")
    
    # Pipeline metadata
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    git_commit: Optional[str] = Field(None, description="Git commit of ingestion code")
    config_hash: Optional[str] = Field(None, description="Hash of ingestion config")
    
    # Embedding metadata
    embedding_model: Optional[str] = Field(None, description="Embedding model used")
    embedding_dim: Optional[int] = Field(None, description="Embedding dimension")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for vector store payload."""
        return self.dict()


class IngestionRecord(BaseModel):
    """Record of an ingestion run for lineage tracking."""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dataset_name: str
    dataset_version: str
    config_hash: str
    git_commit: Optional[str]
    
    # Counts
    total_documents: int
    total_chunks: int
    successful_chunks: int
    failed_chunks: int
    
    # Timing
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # Sample IDs for verification
    sample_doc_ids: List[str] = Field(default_factory=list)
    sample_chunk_ids: List[str] = Field(default_factory=list)
    
    # Configuration
    chunk_strategy: Dict[str, Any] = Field(default_factory=dict)
    embedding_strategy: Dict[str, Any] = Field(default_factory=dict)
    
    def mark_complete(self):
        """Mark the ingestion as completed."""
        self.completed_at = datetime.utcnow()


def normalize_text(text: str) -> str:
    """Normalize text for consistent hashing."""
    return " ".join(text.strip().split())


def compute_content_hash(text: str) -> str:
    """Compute SHA256 hash of normalized text."""
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def build_doc_id(source: str, external_id: str, content_hash: str) -> str:
    """Build deterministic document ID."""
    return f"{source}:{external_id}:{content_hash[:12]}"


def build_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Build deterministic chunk ID."""
    return f"{doc_id}#c{chunk_index:04d}"


class DatasetAdapter(ABC):
    """Abstract adapter interface for dataset-specific processing."""
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the source/dataset name."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Return the dataset version."""
        pass
    
    @abstractmethod
    def read_rows(self, split: DatasetSplit = DatasetSplit.ALL) -> Iterable[BaseRow]:
        """Read raw dataset rows."""
        pass
    
    @abstractmethod
    def to_documents(self, rows: List[BaseRow], split: DatasetSplit) -> List[Document]:
        """Convert rows to LangChain Documents with metadata."""
        pass
    
    @abstractmethod
    def get_evaluation_queries(self, split: DatasetSplit = DatasetSplit.TEST) -> List[Dict[str, Any]]:
        """Return evaluation queries for this dataset."""
        pass


class ValidationResult(BaseModel):
    """Result of document validation."""
    valid: bool
    doc_id: str
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class SmokeTestResult(BaseModel):
    """Result of post-ingestion smoke tests."""
    passed: bool
    test_name: str
    details: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)


class RetrievalMetrics(BaseModel):
    """Standard retrieval evaluation metrics."""
    recall_at_k: Dict[int, float] = Field(default_factory=dict)
    precision_at_k: Dict[int, float] = Field(default_factory=dict)
    ndcg_at_k: Dict[int, float] = Field(default_factory=dict)
    mrr: float = 0.0
    map_score: float = 0.0
    
    # Additional metrics
    total_queries: int = 0
    total_relevant: int = 0
    
    def add_k_metrics(self, k: int, recall: float, precision: float, ndcg: float):
        """Add metrics for a specific k value."""
        self.recall_at_k[k] = recall
        self.precision_at_k[k] = precision
        self.ndcg_at_k[k] = ndcg


class EvaluationRun(BaseModel):
    """Complete evaluation run results."""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dataset_name: str
    dataset_version: str
    collection_name: str
    
    # Configuration
    retriever_config: Dict[str, Any]
    embedding_config: Dict[str, Any]
    
    # Results
    metrics: RetrievalMetrics
    per_query_results: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Metadata
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)
    git_commit: Optional[str] = None
    
    def save_to_file(self, path: Path):
        """Save evaluation results to JSON file."""
        import json
        
        with open(path, 'w') as f:
            json.dump(
                self.dict(), 
                f, 
                indent=2, 
                default=str  # Handle datetime serialization
            )
