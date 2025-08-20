"""
Adapter for Stack Overflow dataset (SOSum format).
Handles the SOSum dataset from: https://github.com/BonanKou/SOSum-A-Dataset-of-Extractive-Summaries-of-Stack-Overflow-Posts-and-labeling-tools
"""
import json
import csv
import ast
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional

from pipelines.contracts import BaseRow, DatasetAdapter, DatasetSplit
from langchain_core.documents import Document


class StackOverflowRow(BaseRow):
    """Row schema for Stack Overflow posts (SOSum format)."""
    title: str
    body: str  # For questions: question body, for answers: answer body
    tags: List[str] = []
    post_type: str = "question"  # "question" or "answer"
    question_type: Optional[int] = None  # 1=conceptual, 2=how-to, 3=debug-corrective
    summary: Optional[str] = None  # For answers: extractive summary
    related_posts: List[str] = []  # For questions: answer post IDs
    
    class Config:
        extra = "allow"


class StackOverflowAdapter(DatasetAdapter):
    """Adapter for Stack Overflow dataset (SOSum format)."""
    
    def __init__(self, dataset_path: str, version: str = "1.0.0"):
        self.dataset_path = Path(dataset_path)
        self._version = version
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"SOSum dataset not found at {self.dataset_path}")
        
        # Check for SOSum format files
        self.question_file = self.dataset_path / "question.csv"
        self.answer_file = self.dataset_path / "answer.csv"
        
        # Also check for data subfolder (common in SOSum)
        if not self.question_file.exists():
            data_dir = self.dataset_path / "data"
            if data_dir.exists():
                self.question_file = data_dir / "question.csv"
                self.answer_file = data_dir / "answer.csv"
        
        if not self.question_file.exists() or not self.answer_file.exists():
            raise FileNotFoundError(
                f"SOSum format files not found. Expected question.csv and answer.csv in {self.dataset_path} or {self.dataset_path}/data/"
            )
    
    @property
    def source_name(self) -> str:
        return "stackoverflow_sosum"
    
    @property
    def version(self) -> str:
        return self._version
    
    def read_rows(self, split: DatasetSplit = DatasetSplit.ALL) -> Iterable[StackOverflowRow]:
        """Read SOSum Stack Overflow posts from CSV files."""
        # Read questions first
        yield from self._read_questions()
        
        # Then read answers
        yield from self._read_answers()
    
    def _read_questions(self) -> Iterable[StackOverflowRow]:
        """Read questions from question.csv."""
        try:
            with open(self.question_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row_num, row in enumerate(reader):
                    try:
                        yield self._parse_question_row(row, f"q{row_num}")
                    except Exception as e:
                        print(f"Error parsing question row {row_num}: {e}")
                        continue
        except Exception as e:
            print(f"Error reading questions file {self.question_file}: {e}")
    
    def _read_answers(self) -> Iterable[StackOverflowRow]:
        """Read answers from answer.csv."""
        try:
            with open(self.answer_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row_num, row in enumerate(reader):
                    try:
                        yield self._parse_answer_row(row, f"a{row_num}")
                    except Exception as e:
                        print(f"Error parsing answer row {row_num}: {e}")
                        continue
        except Exception as e:
            print(f"Error reading answers file {self.answer_file}: {e}")
    
    def _parse_question_row(self, row: Dict[str, str], external_id: str) -> StackOverflowRow:
        """Parse a question row from question.csv."""
        question_id = row.get("Question Id", row.get("question_id", external_id))
        title = row.get("Question Title", row.get("question_title", ""))
        body = row.get("Question Body", row.get("question_body", ""))
        
        # Parse question body if it's a list representation
        if body.startswith('[') and body.endswith(']'):
            try:
                body_list = ast.literal_eval(body)
                body = " ".join(body_list) if isinstance(body_list, list) else body
            except:
                pass  # Keep original if parsing fails
        
        # Parse tags
        tags_str = row.get("Tags", row.get("tags", ""))
        tags = []
        if tags_str:
            # Handle different tag formats
            if tags_str.startswith('[') and tags_str.endswith(']'):
                try:
                    tags = ast.literal_eval(tags_str)
                except:
                    tags = tags_str.strip('[]').replace("'", "").split(',')
            else:
                tags = tags_str.split(',')
            tags = [tag.strip() for tag in tags if tag.strip()]
        
        # Parse question type
        question_type = None
        type_str = row.get("Question Type", row.get("question_type"))
        if type_str:
            try:
                question_type = int(type_str)
            except:
                pass
        
        # Parse related answer posts
        answer_posts = []
        answer_posts_str = row.get("Answer Posts", row.get("answer_posts", ""))
        if answer_posts_str:
            answer_posts = [post.strip() for post in answer_posts_str.split(',') if post.strip()]
        
        return StackOverflowRow(
            external_id=f"q_{question_id}",
            title=title,
            body=body,
            tags=tags,
            post_type="question",
            question_type=question_type,
            related_posts=answer_posts
        )
    
    def _parse_answer_row(self, row: Dict[str, str], external_id: str) -> StackOverflowRow:
        """Parse an answer row from answer.csv."""
        answer_id = row.get("Answer Id", row.get("answer_id", external_id))
        body = row.get("Answer Body", row.get("answer_body", ""))
        summary = row.get("Summary", row.get("summary", ""))
        
        # Parse answer body if it's a list representation
        if body.startswith('[') and body.endswith(']'):
            try:
                body_list = ast.literal_eval(body)
                body = " ".join(body_list) if isinstance(body_list, list) else body
            except:
                pass
        
        # Parse summary if it's a list representation
        if summary.startswith('[') and summary.endswith(']'):
            try:
                summary_list = ast.literal_eval(summary)
                summary = " ".join(summary_list) if isinstance(summary_list, list) else summary
            except:
                pass
        
        return StackOverflowRow(
            external_id=f"a_{answer_id}",
            title="",  # Answers don't have titles
            body=body,
            tags=[],   # Answers inherit tags from questions
            post_type="answer",
            summary=summary
        )
    def to_documents(self, rows: Iterable[StackOverflowRow]) -> List[Document]:
        """Convert SOSum rows to LangChain documents."""
        documents = []
        
        for row in rows:
            # Create content based on post type
            if row.post_type == "question":
                content = f"Title: {row.title}\n\nQuestion: {row.body}"
                doc_type = "question"
            else:  # answer
                content = row.body
                if row.summary:
                    content = f"Answer: {row.body}\n\nSummary: {row.summary}"
                doc_type = "answer"
            
            # Skip empty content
            if not content.strip():
                continue
            
            metadata = {
                "external_id": row.external_id,
                "source": self.source_name,
                "post_type": row.post_type,
                "doc_type": doc_type,
                "tags": row.tags,
                "title": row.title if row.title else None,
            }
            
            # Add question-specific metadata
            if row.post_type == "question":
                metadata.update({
                    "question_type": row.question_type,
                    "related_posts": row.related_posts,
                })
            
            # Add answer-specific metadata
            if row.post_type == "answer" and row.summary:
                metadata["has_summary"] = True
                metadata["summary"] = row.summary
            
            # Remove None values
            metadata = {k: v for k, v in metadata.items() if v is not None}
            
            documents.append(Document(
                page_content=content,
                metadata=metadata
            ))
        
        return documents
    
    def get_evaluation_queries(self) -> List[Dict[str, Any]]:
        """Get evaluation queries based on question titles and summaries."""
        evaluation_queries = []
        
        # Read questions for evaluation queries
        try:
            with open(self.question_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row_num, row in enumerate(reader):
                    title = row.get("Question Title", row.get("question_title", ""))
                    question_id = row.get("Question Id", row.get("question_id", f"q{row_num}"))
                    
                    if title and len(title) > 10:  # Only use substantial titles
                        # Create query from question title
                        evaluation_queries.append({
                            "query": title,
                            "expected_docs": [f"q_{question_id}"],
                            "query_type": "question_title",
                            "query_id": f"eval_q_{question_id}"
                        })
                        
                        # Also create shortened query (first few words)
                        short_query = " ".join(title.split()[:5])
                        if len(short_query) > 5:
                            evaluation_queries.append({
                                "query": short_query,
                                "expected_docs": [f"q_{question_id}"],
                                "query_type": "question_short",
                                "query_id": f"eval_q_short_{question_id}"
                            })
                    
                    # Limit to reasonable number for testing
                    if len(evaluation_queries) >= 50:
                        break
        except Exception as e:
            print(f"Error creating evaluation queries from questions: {e}")
        
        # Add queries based on answer summaries
        try:
            with open(self.answer_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row_num, row in enumerate(reader):
                    summary = row.get("Summary", row.get("summary", ""))
                    answer_id = row.get("Answer Id", row.get("answer_id", f"a{row_num}"))
                    
                    if summary and len(summary) > 20:  # Only use substantial summaries
                        # Parse summary if it's a list
                        if summary.startswith('[') and summary.endswith(']'):
                            try:
                                summary_list = ast.literal_eval(summary)
                                summary = " ".join(summary_list[:2])  # Use first 2 sentences
                            except:
                                summary = summary[:100]  # Fallback to first 100 chars
                        
                        if len(summary) > 10:
                            evaluation_queries.append({
                                "query": summary,
                                "expected_docs": [f"a_{answer_id}"],
                                "query_type": "answer_summary",
                                "query_id": f"eval_a_{answer_id}"
                            })
                    
                    # Limit total queries
                    if len(evaluation_queries) >= 100:
                        break
        except Exception as e:
            print(f"Error creating evaluation queries from answers: {e}")
        
        return evaluation_queries[:100]  # Return max 100 queries
