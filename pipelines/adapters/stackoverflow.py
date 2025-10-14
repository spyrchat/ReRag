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
    # 1=conceptual, 2=how-to, 3=debug-corrective
    question_type: Optional[int] = None
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
            raise FileNotFoundError(
                f"SOSum dataset not found at {self.dataset_path}")

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
        question_id = row.get(
            "Question Id", row.get("question_id", external_id))
        title = row.get("Question Title", row.get("question_title", ""))
        body = row.get("Question Body", row.get("question_body", ""))

        # Parse question body if it's a list representation
        if body.startswith('[') and body.endswith(']'):
            try:
                body_list = ast.literal_eval(body)
                body = " ".join(body_list) if isinstance(
                    body_list, list) else body
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
            # Handle list format like [315365] or [315365, 123456]
            if answer_posts_str.startswith('[') and answer_posts_str.endswith(']'):
                try:
                    answer_posts_list = ast.literal_eval(answer_posts_str)
                    answer_posts = [str(post).strip()
                                    for post in answer_posts_list if str(post).strip()]
                except:
                    # Fallback to manual parsing
                    clean_str = answer_posts_str.strip('[]')
                    answer_posts = [post.strip()
                                    for post in clean_str.split(',') if post.strip()]
            else:
                answer_posts = [
                    post.strip() for post in answer_posts_str.split(',') if post.strip()]

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
                body = " ".join(body_list) if isinstance(
                    body_list, list) else body
            except:
                pass

        # Parse summary if it's a list representation
        if summary.startswith('[') and summary.endswith(']'):
            try:
                summary_list = ast.literal_eval(summary)
                summary = " ".join(summary_list) if isinstance(
                    summary_list, list) else summary
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

    def to_documents(self, rows: Iterable[StackOverflowRow], split: DatasetSplit = DatasetSplit.ALL) -> List[Document]:
        """Convert SOSum rows to LangChain documents.

        RAG-optimized approach: Only answers are ingested as retrievable documents.
        Questions are stored as metadata/context for their corresponding answers.
        This ensures users retrieve valuable answers, not just questions.
        """
        documents = []
        questions_map = {}  # Map question_id -> question data

        # First pass: collect all rows and build questions map
        all_rows = list(rows)

        # Build questions map for context
        for row in all_rows:
            if row.post_type == "question":
                questions_map[row.external_id] = row

        # Second pass: create documents only from answers, with question context
        for row in all_rows:
            if row.post_type != "answer":
                continue  # Only process answers as primary documents

            # Skip empty answers
            if not row.body.strip():
                continue

            # Find the corresponding question for context
            question_context = None
            question_title = ""
            question_tags = []

            # Extract the numeric answer ID (remove 'a_' prefix)
            answer_id_num = row.external_id.replace('a_', '')

            # Try to find matching question by looking for related posts
            for q_id, question in questions_map.items():
                # Check if this answer ID is in the question's related_posts
                if answer_id_num in question.related_posts:
                    question_context = question.body
                    question_title = question.title
                    question_tags = question.tags
                    break

            # If no direct link found, try to find question with similar ID
            if not question_context:
                for q_id, question in questions_map.items():
                    question_id_num = q_id.replace('q_', '')
                    if question_id_num == answer_id_num:
                        question_context = question.body
                        question_title = question.title
                        question_tags = question.tags
                        break

            content_parts = []

            # Build answer content
            answer_content = row.body.strip()
            content_parts.append(answer_content)

            # Add summary if available
            if row.summary and row.summary.strip():
                content_parts.append(f"Summary: {row.summary.strip()}")

            # Join only answer parts (NO question content)
            content = "\n\n".join(content_parts)

            metadata = {
                "external_id": row.external_id,
                "source": self.source_name,
                "post_type": "answer",
                "doc_type": "answer",
                "tags": question_tags,
                "split": split.value,
                "answer_body": row.body,
                "question_title": question_title if question_title else None,
                "question_context": question_context if question_context else None,
                "has_question_context": bool(question_context),
            }

            # Add answer-specific metadata
            if row.summary:
                metadata["summary"] = row.summary
                metadata["has_summary"] = True

            # Remove None values
            metadata = {k: v for k, v in metadata.items() if v is not None}

            documents.append(Document(
                page_content=content,  # Only answer content
                metadata=metadata     # Question info in metadata
            ))

        return documents

    def get_evaluation_queries(self) -> List[Dict[str, Any]]:
        """Get evaluation queries optimized for answer retrieval.

        Creates queries based on question titles/content that should retrieve 
        the corresponding answers, not the questions themselves.
        """
        evaluation_queries = []

        # Build a map of questions to their related answers
        question_to_answers = {}
        answer_ids = set()

        # First, collect all answer IDs
        try:
            with open(self.answer_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row_num, row in enumerate(reader):
                    answer_id = row.get("Answer Id", row.get(
                        "answer_id", f"a{row_num}"))
                    answer_ids.add(f"a_{answer_id}")
        except Exception as e:
            print(f"Error reading answer IDs: {e}")

        # Read questions and create queries that should retrieve answers
        try:
            with open(self.question_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row_num, row in enumerate(reader):
                    title = row.get("Question Title",
                                    row.get("question_title", ""))
                    body = row.get("Question Body",
                                   row.get("question_body", ""))
                    question_id = row.get("Question Id", row.get(
                        "question_id", f"q{row_num}"))

                    # Parse related answer posts
                    answer_posts_str = row.get(
                        "Answer Posts", row.get("answer_posts", ""))
                    related_answers = []
                    if answer_posts_str:
                        related_answers = [
                            f"a_{post.strip()}" for post in answer_posts_str.split(',') if post.strip()]
                        # Filter to only include answers that actually exist
                        related_answers = [
                            aid for aid in related_answers if aid in answer_ids]

                    # If no explicit related answers, try to infer by ID similarity
                    if not related_answers:
                        potential_answer = f"a_{question_id}"
                        if potential_answer in answer_ids:
                            related_answers = [potential_answer]

                    # Only create queries if we have answers to retrieve
                    if related_answers and title and len(title) > 10:
                        # Query from question title - should retrieve answers
                        evaluation_queries.append({
                            "query": title,
                            "expected_docs": related_answers,
                            "query_type": "question_title_to_answer",
                            "query_id": f"eval_q2a_{question_id}",
                            "description": f"Question title should retrieve answer(s)"
                        })

                        # Query from shortened title
                        short_query = " ".join(title.split()[:6])
                        if len(short_query) > 10:
                            evaluation_queries.append({
                                "query": short_query,
                                "expected_docs": related_answers,
                                "query_type": "question_short_to_answer",
                                "query_id": f"eval_q2a_short_{question_id}",
                                "description": f"Short question should retrieve answer(s)"
                            })

                        # Query from question body (first sentence)
                        if body and len(body) > 20:
                            # Parse body if it's a list
                            if body.startswith('[') and body.endswith(']'):
                                try:
                                    body_list = ast.literal_eval(body)
                                    body = " ".join(body_list) if isinstance(
                                        body_list, list) else body
                                except:
                                    pass

                            # Take first sentence or first 100 chars
                            first_sentence = body.split(
                                '.')[0] if '.' in body else body[:100]
                            if len(first_sentence) > 20:
                                evaluation_queries.append({
                                    "query": first_sentence,
                                    "expected_docs": related_answers,
                                    "query_type": "question_body_to_answer",
                                    "query_id": f"eval_qbody2a_{question_id}",
                                    "description": f"Question body should retrieve answer(s)"
                                })

                    # Limit to reasonable number for testing
                    if len(evaluation_queries) >= 60:
                        break
        except Exception as e:
            print(f"Error creating evaluation queries from questions: {e}")

        # Add queries based on answer summaries - these should retrieve the same answers
        try:
            with open(self.answer_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row_num, row in enumerate(reader):
                    summary = row.get("Summary", row.get("summary", ""))
                    answer_id = row.get("Answer Id", row.get(
                        "answer_id", f"a{row_num}"))
                    expected_answer_id = f"a_{answer_id}"

                    if summary and len(summary) > 20 and expected_answer_id in answer_ids:
                        # Parse summary if it's a list
                        if summary.startswith('[') and summary.endswith(']'):
                            try:
                                summary_list = ast.literal_eval(summary)
                                # Use first 2 sentences
                                summary = " ".join(summary_list[:2])
                            except:
                                # Fallback to first 150 chars
                                summary = summary[:150]

                        if len(summary) > 15:
                            evaluation_queries.append({
                                "query": summary,
                                "expected_docs": [expected_answer_id],
                                "query_type": "answer_summary_to_answer",
                                "query_id": f"eval_a2a_{answer_id}",
                                "description": f"Answer summary should retrieve the full answer"
                            })

                    # Limit total queries
                    if len(evaluation_queries) >= 100:
                        break
        except Exception as e:
            print(f"Error creating evaluation queries from answers: {e}")

        return evaluation_queries[:100]  # Return max 100 queries
