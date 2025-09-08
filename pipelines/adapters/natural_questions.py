"""
Adapter for Natural Questions dataset.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Iterable

from pipelines.contracts import BaseRow, DatasetAdapter, DatasetSplit
from langchain_core.documents import Document


class NaturalQuestionsRow(BaseRow):
    """Row schema for Natural Questions dataset."""
    question: str
    answer: str = ""
    context: str = ""
    long_answer: str = ""
    short_answers: List[str] = []
    
    class Config:
        extra = "allow"


class NaturalQuestionsAdapter(DatasetAdapter):
    """Adapter for Natural Questions dataset."""
    
    def __init__(self, dataset_path: str, version: str = "1.0.0"):
        self.dataset_path = Path(dataset_path)
        self._version = version
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Natural Questions dataset not found at {self.dataset_path}")
    
    @property
    def source_name(self) -> str:
        return "natural_questions"
    
    @property
    def version(self) -> str:
        return self._version
    
    def read_rows(self, split: DatasetSplit = DatasetSplit.ALL) -> Iterable[NaturalQuestionsRow]:
        """Read Natural Questions rows from JSONL files."""
        # Common NQ file patterns
        file_patterns = {
            DatasetSplit.TRAIN: ["train*.jsonl", "nq-train-*.jsonl"],
            DatasetSplit.VALIDATION: ["dev*.jsonl", "nq-dev-*.jsonl", "val*.jsonl"],
            DatasetSplit.TEST: ["test*.jsonl", "nq-test-*.jsonl"]
        }
        
        files_to_read = []
        if split == DatasetSplit.ALL:
            for patterns in file_patterns.values():
                for pattern in patterns:
                    files_to_read.extend(self.dataset_path.glob(pattern))
        else:
            for pattern in file_patterns.get(split, []):
                files_to_read.extend(self.dataset_path.glob(pattern))
        
        # Fallback: read any JSONL files
        if not files_to_read:
            files_to_read = list(self.dataset_path.glob("*.jsonl"))
        
        for file_path in files_to_read:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        data = json.loads(line.strip())
                        yield self._parse_nq_item(data, f"{file_path.name}:{line_num}")
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Skipping malformed line in {file_path}:{line_num}: {e}")
                        continue
    
    def _parse_nq_item(self, data: Dict[str, Any], external_id: str) -> NaturalQuestionsRow:
        """Parse a Natural Questions item."""
        # Handle different NQ formats
        question = data.get("question", data.get("question_text", ""))
        
        # Extract answers - NQ has complex answer structures
        short_answers = []
        long_answer = ""
        
        if "annotations" in data:
            for annotation in data["annotations"]:
                if "short_answers" in annotation:
                    for sa in annotation["short_answers"]:
                        if "text" in sa:
                            short_answers.append(sa["text"])
                
                if "long_answer" in annotation and "candidate_text" in annotation["long_answer"]:
                    long_answer = annotation["long_answer"]["candidate_text"]
        
        # Fallback for simpler formats
        if not short_answers and "answer" in data:
            if isinstance(data["answer"], list):
                short_answers = data["answer"]
            else:
                short_answers = [str(data["answer"])]
        
        context = data.get("document_text", data.get("context", ""))
        
        return NaturalQuestionsRow(
            external_id=external_id,
            question=question,
            answer=short_answers[0] if short_answers else "",
            context=context,
            long_answer=long_answer,
            short_answers=short_answers
        )
    
    def to_documents(self, rows: List[NaturalQuestionsRow], split: DatasetSplit) -> List[Document]:
        """Convert NQ rows to Documents - treating contexts as retrievable documents."""
        documents = []
        
        for row in rows:
            if not row.context or not row.context.strip():
                continue
            
            # Create document from context
            metadata = {
                "external_id": row.external_id,
                "question": row.question,
                "answers": row.short_answers,
                "long_answer": row.long_answer,
                "source": self.source_name,
                "dataset_version": self.version,
                "split": split.value,
                "doc_type": "context"
            }
            
            documents.append(Document(
                page_content=row.context,
                metadata=metadata
            ))
        
        return documents
    
    def get_evaluation_queries(self, split: DatasetSplit = DatasetSplit.TEST) -> List[Dict[str, Any]]:
        """Return evaluation queries for Natural Questions."""
        eval_queries = []
        
        for row in self.read_rows(split):
            if not row.question:
                continue
            
            # For NQ, relevant docs are the contexts that contain answers
            relevant_docs = [row.external_id] if row.context and row.short_answers else []
            
            eval_queries.append({
                "query_id": row.external_id,
                "query": row.question,
                "relevant_doc_ids": relevant_docs,
                "gold_answers": row.short_answers,
                "long_answer": row.long_answer
            })
        
        return eval_queries
