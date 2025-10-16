"""
Ground Truth Generation Script for SOSUM Stack Overflow Dataset

This script:
1. Loads all 506 questions from SOSUM dataset (title + question body)
2. For each question, runs the RAG pipeline (retrieval + generation)
3. Saves results to JSON for LLM-as-judge evaluation

Output JSON structure:
{
    "metadata": {...},
    "questions": [
        {
            "question_id": "...",
            "question_title": "...",
            "question_body": "...",
            "question_full_text": "...",
            "tags": [...],
            "generated_answer": "...",
            "retrieved_context": [
                {"page_content": "...", "score": ..., "metadata": {...}},
                ...
            ],
            "retrieval_metadata": {...},
            "generation_timestamp": "..."
        },
        ...
    ]
}
"""

import json
import csv
import ast
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import logging

# Try to import tqdm, fall back to simple progress if not available
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        """Fallback progress indicator"""
        desc = kwargs.get('desc', '')
        total = len(iterable) if hasattr(iterable, '__len__') else None
        if desc:
            print(f"{desc}...")
        for i, item in enumerate(iterable, 1):
            if total:
                print(f"Progress: {i}/{total}", end='\r')
            yield item
        print()

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.config_loader import load_config
from logs.utils.logger import get_logger

logger = get_logger(__name__)


class GroundTruthGenerator:
    """Generates ground truth answers for SOSUM dataset using the RAG agent."""

    def __init__(
        self,
        dataset_path: str = "datasets/sosum/data",
        output_path: str = "results/ground_truth",
        max_questions: int = None,
        retrieval_top_k: int = 10,
        use_self_rag: bool = True
    ):
        """
        Initialize the ground truth generator.

        Args:
            dataset_path: Path to SOSUM dataset directory
            output_path: Path to save results
            max_questions: Maximum number of questions to process (None = all)
            retrieval_top_k: Number of documents to retrieve per question
            use_self_rag: Use Self-RAG mode (with verification loop) instead of standard RAG
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.max_questions = max_questions
        self.retrieval_top_k = retrieval_top_k
        self.use_self_rag = use_self_rag

        # Import the appropriate graph based on mode
        if use_self_rag:
            from agent.graph_self_rag import graph
            self.mode_name = "self_rag"
            logger.info("Using Self-RAG mode (with verification loop)")
        else:
            from agent.graph_refined import graph
            self.mode_name = "standard"
            logger.info("Using Standard RAG mode")

        self.graph = graph

        # Load config for metadata
        self.config = load_config()

        logger.info(f"Initialized GroundTruthGenerator")
        logger.info(f"Mode: {self.mode_name.upper()}")
        logger.info(f"Dataset path: {self.dataset_path}")
        logger.info(f"Output path: {self.output_path}")
        logger.info(f"Max questions: {self.max_questions or 'all'}")
        logger.info(f"Retrieval top_k: {self.retrieval_top_k}")

    def load_questions(self) -> List[Dict[str, Any]]:
        """
        Load questions from SOSUM question.csv file.

        Returns:
            List of question dictionaries with full text
        """
        questions_file = self.dataset_path / "question.csv"

        if not questions_file.exists():
            raise FileNotFoundError(
                f"Questions file not found: {questions_file}")

        questions = []

        with open(questions_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in tqdm(reader, desc="Loading questions"):
                try:
                    # Parse question_body which is a Python list stored as string
                    question_body_list = ast.literal_eval(row['question_body'])

                    # First item is the title, rest is the body
                    title = question_body_list[0] if question_body_list else ""
                    body_sentences = question_body_list[1:] if len(
                        question_body_list) > 1 else []

                    # Join body sentences
                    body = " ".join(body_sentences)

                    # Create full question text (title + body)
                    full_text = f"{title}\n\n{body}".strip()

                    # Parse tags
                    tags = ast.literal_eval(row['tags']) if row['tags'] else []

                    question = {
                        "question_id": row['question_id'],
                        "question_type": int(row['question_type']),
                        "question_title": title,
                        "question_body": body,
                        "question_full_text": full_text,
                        "tags": tags,
                        "answer_posts": ast.literal_eval(row['answer_posts']) if row['answer_posts'] else []
                    }

                    questions.append(question)

                except Exception as e:
                    logger.error(
                        f"Error parsing question {row.get('question_id', 'unknown')}: {e}")
                    continue

        logger.info(f"Loaded {len(questions)} questions from SOSUM dataset")

        # Limit questions if specified
        if self.max_questions and self.max_questions < len(questions):
            questions = questions[:self.max_questions]
            logger.info(f"Limited to {self.max_questions} questions")

        return questions

    def generate_answer_for_question(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an answer for a single question using the RAG agent.

        Args:
            question: Question dictionary

        Returns:
            Result dictionary with answer and retrieved context
        """
        question_text = question['question_full_text']

        # Prepare agent state
        state = {
            "question": question_text,
            "chat_history": [],
            "top_k": self.retrieval_top_k
        }

        try:
            # Run the agent graph
            final_state = self.graph.invoke(state)

            # Extract retrieved documents - keep only essential info
            retrieved_contexts = []
            if "retrieved_documents" in final_state:
                for doc in final_state["retrieved_documents"]:
                    retrieved_contexts.append(doc.page_content)

            # Combine all context into single string
            context_combined = "\n\n".join(retrieved_contexts)

            result = {
                "question_id": question['question_id'],
                "question": question['question_full_text'],
                "context": context_combined,
                "answer": final_state.get("answer", ""),
                # Optional metadata (can be removed if not needed)
                "metadata": {
                    "question_type": question['question_type'],
                    "tags": question['tags'],
                    "num_retrieved_docs": len(retrieved_contexts),
                    "mode": self.mode_name
                }
            }

            # Add Self-RAG specific metadata if available
            if self.use_self_rag and "self_rag_metadata" in final_state:
                result["metadata"]["self_rag"] = {
                    "iterations": final_state["self_rag_metadata"].get("iterations", 0),
                    "converged": final_state["self_rag_metadata"].get("converged", False),
                    "hallucinations_corrected": final_state["self_rag_metadata"].get("hallucinations_corrected", False)
                }

                # Add verification results if available
                if "verification" in final_state:
                    result["metadata"]["verification"] = {
                        "is_faithful": final_state["verification"].get("is_faithful", None),
                        "confidence": final_state["verification"].get("confidence", None),
                        "severity": final_state["verification"].get("severity", None)
                    }

            return result

        except Exception as e:
            logger.error(
                f"Error generating answer for question {question['question_id']}: {e}")

            return {
                "question_id": question['question_id'],
                "question": question['question_full_text'],
                "context": "",
                "answer": "",
                "metadata": {
                    "question_type": question['question_type'],
                    "tags": question['tags'],
                    "error": str(e)
                }
            }

    def generate_all(self) -> Dict[str, Any]:
        """
        Generate answers for all questions and save results.

        Returns:
            Complete results dictionary
        """
        logger.info("Starting ground truth generation...")

        # Load questions
        questions = self.load_questions()

        # Generate answers
        results = []
        successful = 0
        failed = 0

        for question in tqdm(questions, desc="Generating answers"):
            result = self.generate_answer_for_question(question)
            results.append(result)

            if result.get("error"):
                failed += 1
                logger.warning(
                    f"Failed: {question['question_id']} - {result['error']}")
            else:
                successful += 1

            # Save intermediate results every 10 questions
            if len(results) % 10 == 0:
                self._save_intermediate(results)

        # Create final output
        output = {
            "metadata": {
                "dataset": "sosum",
                "mode": self.mode_name,
                "total_questions": len(questions),
                "successful": successful,
                "failed": failed,
                "retrieval_top_k": self.retrieval_top_k,
                "generated_at": datetime.now().isoformat()
            },
            "data": results
        }

        # Add Self-RAG specific metadata if applicable
        if self.use_self_rag:
            total_iterations = sum(
                r.get("metadata", {}).get("self_rag", {}).get("iterations", 0)
                for r in results
            )
            converged_count = sum(
                1 for r in results
                if r.get("metadata", {}).get("self_rag", {}).get("converged", False)
            )
            corrected_count = sum(
                1 for r in results
                if r.get("metadata", {}).get("self_rag", {}).get("hallucinations_corrected", False)
            )

            output["metadata"]["self_rag_stats"] = {
                "avg_iterations": total_iterations / len(results) if results else 0,
                "converged_count": converged_count,
                "hallucinations_corrected_count": corrected_count
            }

        # Save final results with mode-specific filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_path / \
            f"ground_truth_{self.mode_name}_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Ground truth generation complete!")
        logger.info(f"Total questions: {len(questions)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Output saved to: {output_file}")

        return output

    def _save_intermediate(self, results: List[Dict[str, Any]]):
        """Save intermediate results."""
        intermediate_file = self.output_path / "ground_truth_intermediate.json"
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    """Main entry point for the script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate ground truth answers for SOSUM dataset using RAG agent"
    )
    parser.add_argument(
        "--dataset-path",
        default="datasets/sosum/data",
        help="Path to SOSUM dataset directory (default: datasets/sosum/data)"
    )
    parser.add_argument(
        "--output-path",
        default="results/ground_truth",
        help="Path to save results (default: results/ground_truth)"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions to process (default: all 506)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of documents to retrieve per question (default: 10)"
    )
    parser.add_argument(
        "--use-self-rag",
        action="store_true",
        help="Use Self-RAG mode with verification loop (default: standard RAG)"
    )

    args = parser.parse_args()

    # Print mode info
    mode = "Self-RAG" if args.use_self_rag else "Standard RAG"
    print("\n" + "=" * 70)
    print(f"GROUND TRUTH GENERATION - {mode.upper()} MODE")
    print("=" * 70)
    print(f"Mode: {mode}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_path}")
    print(f"Max questions: {args.max_questions or 'all (506)'}")
    print(f"Top-K retrieval: {args.top_k}")
    if args.use_self_rag:
        print("Self-RAG: Enabled (answers will be verified and refined)")
    print("=" * 70 + "\n")

    # Create generator
    generator = GroundTruthGenerator(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        max_questions=args.max_questions,
        retrieval_top_k=args.top_k,
        use_self_rag=args.use_self_rag
    )

    # Generate ground truth
    results = generator.generate_all()

    print("\n" + "=" * 70)
    print("GROUND TRUTH GENERATION COMPLETE")
    print("=" * 70)
    print(f"Mode: {results['metadata']['mode'].upper()}")
    print(f"Total questions: {results['metadata']['total_questions']}")
    print(f"Successful: {results['metadata']['successful']}")
    print(f"Failed: {results['metadata']['failed']}")

    # Print Self-RAG stats if applicable
    if "self_rag_stats" in results['metadata']:
        stats = results['metadata']['self_rag_stats']
        print(f"\nSelf-RAG Statistics:")
        print(f"  Avg iterations: {stats['avg_iterations']:.2f}")
        print(
            f"  Converged: {stats['converged_count']}/{results['metadata']['successful']}")
        print(
            f"  Hallucinations corrected: {stats['hallucinations_corrected_count']}")

    print(f"\nOutput directory: {generator.output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
