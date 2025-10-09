"""
Generate Baseline Answers WITHOUT RAG

This script generates answers using only the LLM (no retrieval context)
to establish a baseline for comparison with RAG performance.

Output:
- JSON file with questions and LLM-only answers
- Same format as ground_truth_intermediate.json for easy comparison
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
from datetime import datetime
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import load_config
from config.llm_factory import create_llm


class BaselineGenerator:
    """Generate baseline answers without retrieval (no RAG)."""

    def __init__(
        self,
        dataset_path: str = "datasets/sosum/data/question.csv",
        output_path: str = "results/baseline_no_rag",
        max_questions: int = None
    ):
        self.dataset_path = dataset_path
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.max_questions = max_questions

        # Load LLM config
        config = load_config("config.yml")
        self.llm = create_llm(config["llm"])

        print(f"\n{'=' * 70}")
        print("BASELINE GENERATOR (NO RAG)")
        print(f"{'=' * 70}")
        print(f"Dataset: {dataset_path}")
        print(f"Output: {output_path}")
        print(
            f"LLM: {config['llm'].get('provider')}/{config['llm'].get('model')}")
        if max_questions:
            print(f"Limit: {max_questions} questions")
        print(f"{'=' * 70}\n")

    def load_questions(self) -> List[Dict[str, Any]]:
        """Load questions from SOSUM dataset."""
        df = pd.read_csv(self.dataset_path)

        questions = []
        for _, row in df.iterrows():
            # Parse question parts (stored as string representation of list)
            try:
                question_parts = eval(row['question_part'])
                if isinstance(question_parts, list):
                    question_full_text = " ".join(question_parts)
                else:
                    question_full_text = str(question_parts)
            except:
                question_full_text = str(row.get('question_part', ''))

            # Parse tags
            try:
                tags = eval(row['tags']) if pd.notna(row['tags']) else []
            except:
                tags = []

            questions.append({
                'question_id': str(row['question_id']),
                'question_type': int(row['question_type']),
                'title': row['question_title'],
                'question_full_text': question_full_text,
                'tags': tags
            })

        if self.max_questions and self.max_questions < len(questions):
            questions = questions[:self.max_questions]

        print(f"Loaded {len(questions)} questions")
        return questions

    def generate_baseline_answer(self, question: str) -> str:
        """
        Generate answer WITHOUT retrieval context (baseline).

        Args:
            question: The user's question

        Returns:
            Generated answer (no context provided)
        """
        # Prompt without any context - pure LLM knowledge
        prompt = f"""You are an expert software engineer. Answer the following programming question using your knowledge.

Question:
{question}

Provide a clear, helpful answer:"""

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"Error: {str(e)}"

    def generate_all(self) -> Dict[str, Any]:
        """Generate baseline answers for all questions."""
        questions = self.load_questions()

        results = []
        successful = 0
        failed = 0

        print("\nGenerating baseline answers (no RAG)...\n")

        for question in tqdm(questions, desc="Processing"):
            try:
                answer = self.generate_baseline_answer(
                    question['question_full_text'])

                result = {
                    "question_id": question['question_id'],
                    "question": question['question_full_text'],
                    "context": "",  # No context for baseline
                    "answer": answer,
                    "metadata": {
                        "question_type": question['question_type'],
                        "tags": question['tags'],
                        "method": "no_rag_baseline"
                    }
                }

                results.append(result)
                successful += 1

            except Exception as e:
                print(
                    f"\nError processing question {question['question_id']}: {e}")
                results.append({
                    "question_id": question['question_id'],
                    "question": question['question_full_text'],
                    "context": "",
                    "answer": "",
                    "metadata": {
                        "question_type": question['question_type'],
                        "tags": question['tags'],
                        "error": str(e),
                        "method": "no_rag_baseline"
                    }
                })
                failed += 1

            # Auto-save every 10 questions
            if len(results) % 10 == 0:
                self._save_intermediate(results)

        # Save final results
        output_file = self.output_path / \
            f"baseline_no_rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n{'=' * 70}")
        print("BASELINE GENERATION COMPLETE")
        print(f"{'=' * 70}")
        print(f"Total questions: {len(questions)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Output: {output_file}")
        print(f"{'=' * 70}\n")

        return {
            "total": len(questions),
            "successful": successful,
            "failed": failed,
            "results": results,
            "output_file": str(output_file)
        }

    def _save_intermediate(self, results: List[Dict[str, Any]]):
        """Save intermediate results."""
        intermediate_file = self.output_path / "baseline_no_rag_intermediate.json"
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate baseline answers without RAG")
    parser.add_argument(
        "--dataset",
        default="datasets/sosum/data/question.csv",
        help="Path to dataset CSV"
    )
    parser.add_argument(
        "--output",
        default="results/baseline_no_rag",
        help="Output directory"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions to process (default: all)"
    )

    args = parser.parse_args()

    generator = BaselineGenerator(
        dataset_path=args.dataset,
        output_path=args.output,
        max_questions=args.max_questions
    )

    generator.generate_all()


if __name__ == "__main__":
    main()
