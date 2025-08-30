"""StackOverflow benchmark adapter."""
from benchmarks.benchmark_contracts import BenchmarkAdapter, BenchmarkTask, BenchmarkQuery
from typing import List, Union, Dict


class StackOverflowBenchmarkAdapter(BenchmarkAdapter):
    """Benchmark adapter for StackOverflow datasets."""

    @property
    def name(self) -> str:
        return "stackoverflow"

    @property
    def tasks(self) -> List[BenchmarkTask]:
        return [BenchmarkTask.RETRIEVAL, BenchmarkTask.END_TO_END]

    def load_queries(self, split: str = "test") -> List[BenchmarkQuery]:
        """Convert SO questions into benchmark queries."""
        queries = []

        # Strategy 1: Use question titles as queries, full questions as expected context
        for question in self._load_questions(split):
            query = BenchmarkQuery(
                query_id=f"so_{question['question_id']}",
                query_text=question['question_title'],
                expected_answer=self._format_best_answer(question),
                relevant_doc_ids=self._get_related_answer_ids(question),
                category=self._extract_primary_tag(question['tags']),
                difficulty=self._assess_difficulty(question),
                metadata={
                    "original_question": question['question_body'],
                    "tags": question['tags'],
                    "answer_count": len(question.get('answers', []))
                }
            )
            queries.append(query)

        return queries

    def _assess_difficulty(self, question: Dict) -> str:
        """Assess question difficulty based on various factors."""
        # Simple heuristic - can be made more sophisticated
        answer_count = len(question.get('answers', []))
        tag_count = len(question.get('tags', []))

        if answer_count <= 1 or tag_count >= 4:
            return "hard"
        elif answer_count <= 3 or tag_count >= 3:
            return "medium"
        else:
            return "easy"
