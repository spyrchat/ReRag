"""StackOverflow benchmark adapter."""
import json
import os
from pathlib import Path
from typing import List, Union, Dict, Any
from .benchmark_contracts import BenchmarkAdapter, BenchmarkTask, BenchmarkQuery


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


class FullDatasetAdapter(StackOverflowBenchmarkAdapter):
    """Adapter that uses the full dataset with ground truth for proper evaluation."""

    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)

    @property
    def name(self) -> str:
        return "stackoverflow_full_dataset"

    def load_queries(self, split: str = "test") -> List[BenchmarkQuery]:
        """Load queries with ground truth from the full dataset."""
        import pandas as pd
        import ast

        question_file = self.dataset_path / "question.csv"

        if not question_file.exists():
            print(f"❌ Question file not found: {question_file}")
            return self._create_dummy_queries()

        try:
            print(f"📂 Loading questions from {question_file}")
            df = pd.read_csv(question_file)
            print(f"📊 Total questions in dataset: {len(df)}")

            # Filter for questions with ground truth (answer_posts)
            df_with_gt = df[df['answer_posts'].notna()]
            print(f"📊 Questions with ground truth: {len(df_with_gt)}")

            queries = []
            for idx, row in df_with_gt.iterrows():
                if pd.isna(row['question_title']) or not row['question_title'].strip():
                    continue

                # Parse answer IDs from the answer_posts field
                try:
                    if isinstance(row['answer_posts'], str):
                        # Try to parse as literal (list format)
                        answer_ids = ast.literal_eval(row['answer_posts'])
                    else:
                        # Could be a single ID or other format
                        answer_ids = [int(row['answer_posts'])]

                    # Convert to document IDs with 'a_' prefix
                    relevant_doc_ids = [f"a_{aid}" for aid in answer_ids]

                    if not relevant_doc_ids:
                        continue  # Skip if no valid answer IDs

                except (ValueError, SyntaxError, TypeError) as e:
                    print(
                        f"⚠️  Failed to parse answer_posts for question {row['question_id']}: {e}")
                    continue

                query = BenchmarkQuery(
                    query_id=f"full_so_{row['question_id']}",
                    query_text=str(row['question_title']).strip(),
                    expected_answer=None,  # We don't need the answer text for retrieval eval
                    relevant_doc_ids=relevant_doc_ids,
                    difficulty="medium",
                    category="programming",
                    metadata={
                        "source": "full_dataset_with_ground_truth",
                        "original_question_id": row['question_id'],
                        "question_type": row.get('question_type', 'unknown'),
                        "tags": row.get('tags', ''),
                        "num_ground_truth_docs": len(relevant_doc_ids)
                    }
                )
                queries.append(query)

            print(
                f"✅ Successfully loaded {len(queries)} queries with ground truth")
            return queries

        except Exception as e:
            print(f"❌ Error loading full dataset: {e}")
            import traceback
            traceback.print_exc()
            return self._create_dummy_queries()

    def get_ground_truth(self, query_id: str) -> Dict[str, Any]:
        """Get ground truth for evaluation (override parent method)."""
        # For this adapter, ground truth is already in the query's relevant_doc_ids
        return {"relevant_docs": [], "expected_answer": None}
