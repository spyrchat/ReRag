"""StackOverflow benchmark adapter."""
import json
import os
from pathlib import Path
from typing import List, Union, Dict, Any
from benchmark_contracts import BenchmarkAdapter, BenchmarkTask, BenchmarkQuery


class StackOverflowBenchmarkAdapter(BenchmarkAdapter):
    """Benchmark adapter for StackOverflow datasets."""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

    @property
    def name(self) -> str:
        return "stackoverflow"

    @property
    def tasks(self) -> List[BenchmarkTask]:
        return [BenchmarkTask.RETRIEVAL, BenchmarkTask.END_TO_END]

    def load_queries(self, split: str = "test") -> List[BenchmarkQuery]:
        """Convert SO questions into benchmark queries."""
        queries = []

        # Try to find CSV or JSON files in the dataset directory
        csv_files = list(self.dataset_path.glob("*.csv"))
        json_files = list(self.dataset_path.glob("*.json"))

        if csv_files:
            queries = self._load_from_csv(csv_files[0])
        elif json_files:
            queries = self._load_from_json(json_files[0])
        else:
            print(f"⚠️  No CSV or JSON files found in {self.dataset_path}")
            # Return dummy queries for testing
            return self._create_dummy_queries()

        print(f"✅ Loaded {len(queries)} queries from {split} split")
        return queries[:100]  # Limit for testing

    def _load_from_csv(self, csv_file: Path) -> List[BenchmarkQuery]:
        """Load queries from CSV file."""
        import pandas as pd

        try:
            df = pd.read_csv(csv_file)
            queries = []

            # Try different column name combinations
            title_col = None
            body_col = None
            id_col = None

            for col in df.columns:
                if 'title' in col.lower() or 'question' in col.lower():
                    title_col = col
                elif 'body' in col.lower() or 'text' in col.lower():
                    body_col = col
                elif 'id' in col.lower():
                    id_col = col

            if not title_col:
                print(
                    f"❌ No title column found. Available columns: {list(df.columns)}")
                return self._create_dummy_queries()

            for idx, row in df.iterrows():
                if idx >= 100:  # Limit for testing
                    break

                query_id = str(row[id_col]) if id_col else f"csv_{idx}"
                title = str(row[title_col])
                body = str(row[body_col]) if body_col else ""

                if not title or title == 'nan':
                    continue

                query = BenchmarkQuery(
                    query_id=f"so_{query_id}",
                    query_text=title,
                    expected_answer=body[:500] if body and body != 'nan' else None,
                    relevant_doc_ids=None,
                    difficulty="medium",
                    category="programming",
                    metadata={
                        "source": "stackoverflow_csv",
                        "row_index": idx
                    }
                )
                queries.append(query)

            return queries

        except Exception as e:
            print(f"❌ Error loading CSV {csv_file}: {e}")
            return self._create_dummy_queries()

    def _load_from_json(self, json_file: Path) -> List[BenchmarkQuery]:
        """Load queries from JSON file."""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            queries = []

            # Handle different JSON structures
            if isinstance(data, list):
                questions = data
            elif isinstance(data, dict) and 'questions' in data:
                questions = data['questions']
            else:
                questions = [data]  # Single question

            for i, question in enumerate(questions[:100]):  # Limit for testing
                query = self._create_query_from_question(question, i)
                if query:
                    queries.append(query)

            return queries

        except Exception as e:
            print(f"❌ Error loading JSON {json_file}: {e}")
            return self._create_dummy_queries()

    def _create_query_from_question(self, question: Dict[str, Any], index: int) -> BenchmarkQuery:
        """Create a benchmark query from a question."""

        # Try different possible field names
        title = question.get('title') or question.get(
            'question_title') or question.get('Title')
        body = question.get('body') or question.get(
            'question_body') or question.get('Body') or ""
        qid = question.get('id') or question.get(
            'question_id') or question.get('Id') or f"q_{index}"

        if not title:
            return None

        return BenchmarkQuery(
            query_id=f"so_{qid}",
            query_text=title,
            expected_answer=body[:500] if body else None,
            relevant_doc_ids=None,
            difficulty="medium",
            category="programming",
            metadata={
                "original_question": question,
                "source": "stackoverflow_json"
            }
        )

    def _create_dummy_queries(self) -> List[BenchmarkQuery]:
        """Create dummy queries for testing."""
        dummy_questions = [
            "How to show error message box in .NET?",
            "What is the difference between StringBuilder and String in C#?",
            "How to convert string to int in Java?",
            "How to handle null values in Python?",
            "What is the best way to iterate over a dictionary in Python?",
            "How to reverse a string in Python?",
            "What is object-oriented programming?",
            "How to use lambda functions in Python?",
            "What is the difference between list and tuple?",
            "How to handle exceptions in Python?"
        ]

        queries = []
        for i, question in enumerate(dummy_questions):
            query = BenchmarkQuery(
                query_id=f"dummy_{i}",
                query_text=question,
                expected_answer=f"Programming answer for: {question}",
                relevant_doc_ids=None,
                difficulty="easy",
                category="programming",
                metadata={"source": "dummy"}
            )
            queries.append(query)

        return queries

    def get_ground_truth(self, query_id: str) -> Dict[str, Any]:
        """Get ground truth for evaluation."""
        return {"relevant_docs": [], "expected_answer": None}
