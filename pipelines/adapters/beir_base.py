"""
Base adapter functionality for BEIR datasets.
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Iterable
from beir.datasets.data_loader import GenericDataLoader

from pipelines.contracts import BaseRow, DatasetAdapter, DatasetSplit
from langchain_core.documents import Document


class BeirBaseAdapter(DatasetAdapter):
    """Base adapter for BEIR datasets."""
    
    def __init__(self, dataset_path: str, dataset_name: str, version: str = "1.0.0"):
        self.dataset_path = Path(dataset_path)
        self.dataset_name = dataset_name
        self._version = version
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
    
    @property
    def source_name(self) -> str:
        return self.dataset_name
    
    @property
    def version(self) -> str:
        return self._version
    
    def _load_beir_data(self, split: DatasetSplit):
        """Load BEIR dataset components."""
        split_name = "test" if split in [DatasetSplit.TEST, DatasetSplit.ALL] else split.value
        
        try:
            corpus, queries, qrels = GenericDataLoader(
                str(self.dataset_path)
            ).load(split=split_name)
            return corpus, queries, qrels
        except Exception as e:
            # Fallback for datasets without train/val splits
            corpus, queries, qrels = GenericDataLoader(
                str(self.dataset_path)
            ).load(split="test")
            return corpus, queries, qrels
    
    def get_evaluation_queries(self, split: DatasetSplit = DatasetSplit.TEST) -> List[Dict[str, Any]]:
        """Return evaluation queries with relevance judgments."""
        _, queries, qrels = self._load_beir_data(split)
        
        eval_queries = []
        for qid, query_text in queries.items():
            relevant_docs = list(qrels.get(qid, {}).keys()) if qid in qrels else []
            
            eval_queries.append({
                "query_id": qid,
                "query": query_text,
                "relevant_doc_ids": relevant_docs,
                "relevance_scores": qrels.get(qid, {})
            })
        
        return eval_queries


class BeirRow(BaseRow):
    """Row schema for BEIR datasets."""
    title: str = ""
    text: str = ""
    metadata: Dict[str, Any] = {}


class GenericBeirAdapter(BeirBaseAdapter):
    """Generic adapter for any BEIR dataset."""
    
    def read_rows(self, split: DatasetSplit = DatasetSplit.ALL) -> Iterable[BeirRow]:
        """Read corpus documents as rows."""
        corpus, _, _ = self._load_beir_data(split)
        
        for doc_id, content in corpus.items():
            yield BeirRow(
                external_id=doc_id,
                title=content.get("title", ""),
                text=content.get("text", ""),
                metadata=content.get("metadata", {})
            )
    
    def to_documents(self, rows: List[BeirRow], split: DatasetSplit) -> List[Document]:
        """Convert rows to LangChain Documents."""
        documents = []
        
        for row in rows:
            # Combine title and text
            content_parts = []
            if row.title:
                content_parts.append(row.title)
            if row.text:
                content_parts.append(row.text)
            
            full_text = ". ".join(content_parts).strip()
            if not full_text:
                continue
            
            metadata = {
                "external_id": row.external_id,
                "title": row.title,
                "source": self.source_name,
                "dataset_version": self.version,
                "split": split.value,
                **row.metadata
            }
            
            documents.append(Document(
                page_content=full_text,
                metadata=metadata
            ))
        
        return documents
