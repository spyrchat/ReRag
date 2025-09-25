"""StackOverflow benchmark adapter."""
import json
import os
from pathlib import Path
from typing import List, Union, Dict, Any
from .benchmark_contracts import BenchmarkAdapter, BenchmarkTask, BenchmarkQuery


class StackOverflowBenchmarkAdapter(BenchmarkAdapter):
    """Benchmark adapter that loads questions with ground truth mappings."""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self._queries_cache = None  # Cache loaded queries

    @property
    def name(self) -> str:
        return "stackoverflow"

    @property
    def tasks(self) -> List[BenchmarkTask]:
        return [BenchmarkTask.RETRIEVAL, BenchmarkTask.END_TO_END]

    def load_queries(self, split: str = "test") -> List[BenchmarkQuery]:
        """Load questions with their corresponding answer document IDs."""
        if self._queries_cache is None:
            # Look for the questions file with ground truth
            question_file = self.dataset_path / "question.csv"

            if not question_file.exists():
                print(f"Question file not found: {question_file}")
                print(
                    f"   Available files: {list(self.dataset_path.glob('*.csv'))}")
                self._queries_cache = self._create_dummy_queries()
            else:
                self._queries_cache = self._load_questions_with_ground_truth(
                    question_file)

        return self._queries_cache

    def get_ground_truth(self, query_id: str) -> List[str]:
        """Get ground truth document IDs for a specific query."""
        queries = self.load_queries()  # This will use cache if already loaded

        for query in queries:
            if query.query_id == query_id:
                return query.relevant_doc_ids or []

        print(f"No ground truth found for query_id: {query_id}")
        return []

    def _load_questions_with_ground_truth(self, question_file: Path) -> List[BenchmarkQuery]:
        """Load questions and map them to their relevant answer document IDs."""
        import pandas as pd
        import ast

        try:
            df = pd.read_csv(question_file)
            print(f"📊 Loaded {len(df)} questions from dataset")
            print(f"📊 Columns: {list(df.columns)}")

            # Filter for questions that have ground truth answers
            df_with_answers = df[df['answer_posts'].notna()]
            print(
                f"📊 Questions with ground truth answers: {len(df_with_answers)}")

            queries = []

            for idx, row in df_with_answers.iterrows():
                # Skip if no question title
                if pd.isna(row['question_title']) or not str(row['question_title']).strip():
                    continue

                # Parse the answer IDs from answer_posts column
                try:
                    answer_posts = row['answer_posts']

                    if isinstance(answer_posts, str):
                        # Handle list format: "[123, 456, 789]"
                        if answer_posts.startswith('[') and answer_posts.endswith(']'):
                            answer_ids = ast.literal_eval(answer_posts)
                        else:
                            # Single ID as string
                            answer_ids = [int(answer_posts)]
                    else:
                        # Single numeric ID
                        answer_ids = [int(answer_posts)]

                    # Convert to string IDs (without 'a_' prefix for ground truth)
                    # The benchmark runner will compare these against extracted IDs
                    relevant_doc_ids = [str(aid) for aid in answer_ids]

                    if not relevant_doc_ids:
                        continue

                except (ValueError, SyntaxError, TypeError) as e:
                    print(
                        f"Failed to parse answer_posts for question {row['question_id']}: {e}")
                    continue

                # Create the benchmark query
                query = BenchmarkQuery(
                    query_id=str(row['question_id']),
                    query_text=str(row['question_title']).strip(),
                    expected_answer=None,  # Not needed for retrieval evaluation
                    # Ground truth: which answers should be retrieved
                    relevant_doc_ids=relevant_doc_ids,
                    difficulty="medium",
                    category="programming",
                    metadata={
                        "source": "stackoverflow_sosum",
                        "question_type": row.get('question_type', 'unknown'),
                        "tags": row.get('tags', ''),
                        "num_answers": len(relevant_doc_ids)
                    }
                )
                queries.append(query)

            print(
                f"Successfully loaded {len(queries)} queries with ground truth")

            # Show sample for debugging
            if queries:
                sample = queries[0]
                print(f"📝 Sample query:")
                print(f"   ID: {sample.query_id}")
                print(f"   Text: {sample.query_text}")
                print(f"   Ground truth docs: {sample.relevant_doc_ids}")

            return queries

        except Exception as e:
            print(f"Error loading questions: {e}")
            import traceback
            traceback.print_exc()
            return self._create_dummy_queries()

    def _create_dummy_queries(self) -> List[BenchmarkQuery]:
        """Fallback dummy queries for testing."""
        return [
            BenchmarkQuery(
                query_id="dummy_1",
                query_text="How to implement binary search tree?",
                expected_answer=None,
                relevant_doc_ids=None,  # No ground truth for dummies
                difficulty="medium",
                category="programming",
                metadata={"source": "dummy"}
            ),
            BenchmarkQuery(
                query_id="dummy_2",
                query_text="How to optimize database queries?",
                expected_answer=None,
                relevant_doc_ids=None,
                difficulty="medium",
                category="programming",
                metadata={"source": "dummy"}
            )
        ]
