"""StackOverflow benchmark adapter."""
import json
import os
from pathlib import Path
from typing import List, Union, Dict, Any
from abc import ABC, abstractmethod
from .benchmark_contracts import BenchmarkTask, BenchmarkQuery
from utils import preload_chunk_id_mapping


class BenchmarkAdapter(ABC):
    """
    Abstract base class for all benchmark adapters.
    Defines the required interface for dataset adapters.
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the dataset/adapter."""
        pass

    @property
    @abstractmethod
    def tasks(self) -> List:
        """List of supported benchmark tasks."""
        pass

    @abstractmethod
    def load_queries(self, split: str = "test") -> List:
        """Load queries for the given split (e.g., 'test', 'train')."""
        pass

    @abstractmethod
    def get_ground_truth(self, query_id: str) -> List[str]:
        """Get ground truth document IDs for a specific query."""
        pass


class StackOverflowBenchmarkAdapter(BenchmarkAdapter):
    """Benchmark adapter that loads questions with ground truth mappings."""

    def __init__(self, dataset_path: str, qdrant_client=None, collection_name=None):
        self.dataset_path = Path(dataset_path)
        self._queries_cache = None  # Cache loaded queries
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name

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
        import pandas as pd
        import ast

        # Preload all chunk IDs for the collection
        chunk_id_mapping = preload_chunk_id_mapping(
            self.qdrant_client, self.collection_name)

        df = pd.read_csv(question_file)
        df_with_answers = df[df['answer_posts'].notna()]
        queries = []

        for idx, row in df_with_answers.iterrows():
            if pd.isna(row['question_title']) or not str(row['question_title']).strip():
                continue

            try:
                answer_posts = row['answer_posts']
                if isinstance(answer_posts, str):
                    if answer_posts.startswith('[') and answer_posts.endswith(']'):
                        answer_ids = ast.literal_eval(answer_posts)
                    else:
                        answer_ids = [int(answer_posts)]
                else:
                    answer_ids = [int(answer_posts)]
                if not answer_ids:
                    continue
            except (ValueError, SyntaxError, TypeError) as e:
                print(
                    f"Failed to parse answer_posts for question {row['question_id']}: {e}")
                continue

            relevant_chunk_ids = []
            for aid in answer_ids:
                external_id = f"a_{aid}"
                chunk_ids = chunk_id_mapping.get(external_id, [])
                relevant_chunk_ids.extend(chunk_ids)

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
                    query_text=f"{row['question_title']} {row.get('question_body', '')}".strip(
                    ),
                    expected_answer=None,  # Not needed for retrieval evaluation
                    # Ground truth: which answers should be retrieved
                    relevant_doc_ids=relevant_chunk_ids,
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
            print("Sample ground truth chunk IDs:", relevant_chunk_ids[:5])
            print(
                f"Successfully loaded {len(queries)} queries with ground truth")

            # # Show sample for debugging
            # if queries:
            #     sample = queries[0]
            #     print(f"📝 Sample query:")
            #     print(f"   ID: {sample.query_id}")
            #     print(f"   Text: {sample.query_text}")
            #     print(f"   Ground truth docs: {sample.relevant_doc_ids}")

        return queries

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
