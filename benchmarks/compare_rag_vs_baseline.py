"""
RAG vs Baseline Comparison Evaluator

This script compares RAG answers against baseline (no-RAG) answers using LLM-as-a-Judge.
It evaluates both systems on the same questions and provides statistical comparison.

Output:
- Side-by-side evaluation scores
- Statistical significance tests
- Performance improvement metrics
- Detailed comparison report
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from scipy import stats

# Make sure this path matches your project
from llm_judge import LLMJudge

# === Config ===
PROVIDER = "openai"       # openai | anthropic
MODEL_NAME = "gpt-4o-mini"  # Judge model

# Input paths
RAG_ANSWERS_PATH = "results/ground_truth_for_test/ground_truth_intermediate.json"
BASELINE_ANSWERS_PATH = "results/baseline_no_rag/baseline_no_rag_intermediate.json"

# Output paths
OUTPUT_DIR = Path("results/rag_vs_baseline_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 20
SLEEP_BETWEEN_BATCHES = 0

# === Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Initialize judge
judge = LLMJudge(provider=PROVIDER, model_name=MODEL_NAME)


def build_evaluation_prompt(question: str, answer: str, context: str, system_type: str) -> str:
    """
    Build evaluation prompt for LLM judge.

    Args:
        question: User's question
        answer: System's answer
        context: Retrieved context (empty for baseline)
        system_type: "RAG" or "Baseline"
    """
    context_note = "Retrieved context was used." if context else "No context was used (baseline)."

    return f"""
You are evaluating a {system_type} system's answer to a software engineering question.

{context_note}

### Evaluation Criteria

Rate each dimension from 1 (poor) to 5 (excellent):

1. **Correctness**: Is the answer technically accurate?
   - 5: Completely correct
   - 3: Partially correct
   - 1: Incorrect or misleading

2. **Completeness**: Does it fully answer the question?
   - 5: Comprehensive answer
   - 3: Partial answer
   - 1: Incomplete or missing key details

3. **Clarity**: Is it easy to understand?
   - 5: Very clear and well-structured
   - 3: Understandable but could be clearer
   - 1: Confusing or poorly explained

4. **Usefulness**: Can the user apply this answer?
   - 5: Immediately actionable
   - 3: Somewhat helpful
   - 1: Not useful

5. **Code Quality** (if applicable): Are code examples good?
   - 5: Excellent, runnable examples
   - 3: Basic examples
   - 1: No examples or poor quality
   - 0: Not applicable (no code needed)

### Format

Return ONLY valid JSON:

{{
  "correctness": 4,
  "completeness": 5,
  "clarity": 4,
  "usefulness": 5,
  "code_quality": 4,
  "justification": "Brief explanation"
}}

### Question
{question}

### Answer
{answer}

{f"### Context (Retrieved Documents)\\n{context}" if context else "### No Context (Baseline - using only LLM knowledge)"}
""".strip()


def evaluate_answer(question: str, answer: str, context: str, system_type: str) -> Dict:
    """Evaluate a single answer with retry logic."""
    max_retries = 3

    for retry in range(max_retries):
        try:
            prompt = build_evaluation_prompt(
                question, answer, context, system_type)
            scores = judge.evaluate(prompt)

            # Validate
            required = ["correctness", "completeness", "clarity", "usefulness"]
            if not all(k in scores for k in required):
                raise ValueError(f"Missing required fields: {scores}")

            # Validate ranges
            for key in required + ["code_quality"]:
                if key in scores and not (0 <= scores[key] <= 5):
                    raise ValueError(f"{key} out of range: {scores[key]}")

            return scores

        except Exception as e:
            if retry < max_retries - 1:
                logger.warning(
                    f"Evaluation failed (attempt {retry + 1}/{max_retries}): {str(e)[:100]}")
                time.sleep(1)
            else:
                logger.error(f"Evaluation failed after {max_retries} attempts")
                return {
                    "correctness": 0,
                    "completeness": 0,
                    "clarity": 0,
                    "usefulness": 0,
                    "code_quality": 0,
                    "justification": f"Evaluation failed: {str(e)[:100]}"
                }


def load_answers(rag_path: str, baseline_path: str) -> Tuple[List[Dict], List[Dict]]:
    """Load RAG and baseline answers."""
    print(f"Loading RAG answers from: {rag_path}")
    with open(rag_path, 'r', encoding='utf-8') as f:
        rag_data = json.load(f)

    print(f"Loading baseline answers from: {baseline_path}")
    with open(baseline_path, 'r', encoding='utf-8') as f:
        baseline_data = json.load(f)

    print(
        f"Loaded {len(rag_data)} RAG answers, {len(baseline_data)} baseline answers")

    # Match questions by question_id
    rag_dict = {item['question_id']: item for item in rag_data}
    baseline_dict = {item['question_id']: item for item in baseline_data}

    # Find common questions
    common_ids = set(rag_dict.keys()) & set(baseline_dict.keys())
    print(f"Found {len(common_ids)} common questions for comparison")

    matched_rag = [rag_dict[qid] for qid in sorted(common_ids)]
    matched_baseline = [baseline_dict[qid] for qid in sorted(common_ids)]

    return matched_rag, matched_baseline


def run_comparison_evaluation():
    """Run comparative evaluation of RAG vs baseline."""
    print(f"\n{'=' * 70}")
    print("RAG vs BASELINE COMPARISON EVALUATION")
    print(f"{'=' * 70}\n")

    # Load data
    rag_answers, baseline_answers = load_answers(
        RAG_ANSWERS_PATH, BASELINE_ANSWERS_PATH)

    results = []

    print(f"\nEvaluating {len(rag_answers)} question pairs...\n")

    for i in tqdm(range(0, len(rag_answers), BATCH_SIZE), desc="Evaluating"):
        batch_rag = rag_answers[i:i + BATCH_SIZE]
        batch_baseline = baseline_answers[i:i + BATCH_SIZE]

        for rag_item, baseline_item in zip(batch_rag, batch_baseline):
            question_id = rag_item['question_id']
            question = rag_item['question']

            # Evaluate RAG answer
            rag_scores = evaluate_answer(
                question=question,
                answer=rag_item['answer'],
                context=rag_item.get('context', ''),
                system_type="RAG"
            )

            # Evaluate baseline answer
            baseline_scores = evaluate_answer(
                question=question,
                answer=baseline_item['answer'],
                context="",
                system_type="Baseline"
            )

            # Calculate improvement
            improvements = {}
            for metric in ["correctness", "completeness", "clarity", "usefulness", "code_quality"]:
                rag_val = rag_scores.get(metric, 0)
                baseline_val = baseline_scores.get(metric, 0)

                if baseline_val > 0:
                    improvements[f"{metric}_improvement_pct"] = (
                        (rag_val - baseline_val) / baseline_val) * 100
                else:
                    improvements[f"{metric}_improvement_pct"] = 0

                improvements[f"{metric}_diff"] = rag_val - baseline_val

            result = {
                "question_id": question_id,
                "question": question,
                "rag_answer": rag_item['answer'],
                "baseline_answer": baseline_item['answer'],
                "rag_scores": rag_scores,
                "baseline_scores": baseline_scores,
                "improvements": improvements,
                "has_rag_context": bool(rag_item.get('context'))
            }

            results.append(result)

        time.sleep(SLEEP_BETWEEN_BATCHES)

    # Save detailed results
    output_file = OUTPUT_DIR / \
        f"comparison_detailed_{PROVIDER}_{MODEL_NAME}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Detailed results saved to: {output_file}")

    # Generate summary statistics
    summary = generate_summary_statistics(results)

    # Save summary
    summary_file = OUTPUT_DIR / \
        f"comparison_summary_{PROVIDER}_{MODEL_NAME}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"✅ Summary saved to: {summary_file}")

    # Print report
    print_comparison_report(summary)

    return results, summary


def generate_summary_statistics(results: List[Dict]) -> Dict:
    """Generate comprehensive statistical summary."""
    metrics = ["correctness", "completeness",
               "clarity", "usefulness", "code_quality"]

    summary = {
        "total_questions": len(results),
        "evaluation_date": pd.Timestamp.now().isoformat(),
        "metrics": {}
    }

    for metric in metrics:
        rag_scores = [r['rag_scores'].get(metric, 0) for r in results]
        baseline_scores = [r['baseline_scores'].get(
            metric, 0) for r in results]
        improvements = [r['improvements'].get(
            f"{metric}_diff", 0) for r in results]

        # Basic statistics
        summary["metrics"][metric] = {
            "rag": {
                "mean": float(np.mean(rag_scores)),
                "std": float(np.std(rag_scores)),
                "median": float(np.median(rag_scores)),
                "min": float(np.min(rag_scores)),
                "max": float(np.max(rag_scores))
            },
            "baseline": {
                "mean": float(np.mean(baseline_scores)),
                "std": float(np.std(baseline_scores)),
                "median": float(np.median(baseline_scores)),
                "min": float(np.min(baseline_scores)),
                "max": float(np.max(baseline_scores))
            },
            "improvement": {
                "mean_diff": float(np.mean(improvements)),
                "mean_pct": float(np.mean([r['improvements'].get(f"{metric}_improvement_pct", 0) for r in results])),
                "median_diff": float(np.median(improvements)),
                "wins": int(sum(1 for i in improvements if i > 0)),
                "ties": int(sum(1 for i in improvements if i == 0)),
                "losses": int(sum(1 for i in improvements if i < 0))
            }
        }

        # Statistical significance (paired t-test)
        if len(rag_scores) > 1:
            t_stat, p_value = stats.ttest_rel(rag_scores, baseline_scores)
            summary["metrics"][metric]["statistical_test"] = {
                "test": "paired_t_test",
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant_at_0.05": bool(p_value < 0.05),
                "significant_at_0.01": bool(p_value < 0.01)
            }

            # Effect size (Cohen's d)
            diff = np.array(rag_scores) - np.array(baseline_scores)
            cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
            summary["metrics"][metric]["effect_size"] = {
                "cohens_d": float(cohens_d),
                "interpretation": interpret_cohens_d(cohens_d)
            }

    # Overall average improvement
    overall_rag = np.mean([np.mean([r['rag_scores'].get(m, 0)
                          for m in metrics]) for r in results])
    overall_baseline = np.mean(
        [np.mean([r['baseline_scores'].get(m, 0) for m in metrics]) for r in results])

    summary["overall"] = {
        "rag_mean": float(overall_rag),
        "baseline_mean": float(overall_baseline),
        "improvement_pct": float(((overall_rag - overall_baseline) / overall_baseline) * 100) if overall_baseline > 0 else 0
    }

    return summary


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def print_comparison_report(summary: Dict):
    """Print formatted comparison report."""
    print(f"\n{'=' * 70}")
    print("RAG vs BASELINE - COMPARISON REPORT")
    print(f"{'=' * 70}\n")

    print(f"Total questions evaluated: {summary['total_questions']}\n")

    print("=" * 70)
    print(f"{'Metric':<20} {'RAG':<12} {'Baseline':<12} {'Improvement':<15} {'Sig.'}")
    print("=" * 70)

    for metric, data in summary['metrics'].items():
        rag_mean = data['rag']['mean']
        baseline_mean = data['baseline']['mean']
        improvement_pct = data['improvement']['mean_pct']

        sig = ""
        if 'statistical_test' in data:
            if data['statistical_test']['significant_at_0.01']:
                sig = "***"
            elif data['statistical_test']['significant_at_0.05']:
                sig = "**"
            else:
                sig = "ns"

        improvement_str = f"+{improvement_pct:>6.1f}%" if improvement_pct > 0 else f"{improvement_pct:>7.1f}%"

        print(f"{metric:<20} {rag_mean:>5.2f} ±{data['rag']['std']:>4.2f}  "
              f"{baseline_mean:>5.2f} ±{data['baseline']['std']:>4.2f}  "
              f"{improvement_str:<15} {sig}")

    print("=" * 70)
    print(f"\n{'Overall Performance':<20} {summary['overall']['rag_mean']:>5.2f}       "
          f"{summary['overall']['baseline_mean']:>5.2f}       "
          f"+{summary['overall']['improvement_pct']:>6.1f}%")
    print("=" * 70)

    print("\nSignificance: *** p<0.01, ** p<0.05, ns = not significant\n")

    # Print effect sizes
    print("Effect Sizes (Cohen's d):")
    print("-" * 70)
    for metric, data in summary['metrics'].items():
        if 'effect_size' in data:
            d = data['effect_size']['cohens_d']
            interpretation = data['effect_size']['interpretation']
            print(f"  {metric:<20} d={d:>6.3f} ({interpretation})")
    print()

    # Win/Tie/Loss summary
    print("Win/Tie/Loss Summary:")
    print("-" * 70)
    for metric, data in summary['metrics'].items():
        wins = data['improvement']['wins']
        ties = data['improvement']['ties']
        losses = data['improvement']['losses']
        total = wins + ties + losses
        print(f"  {metric:<20} W:{wins:>3} ({wins / total * 100:>5.1f}%)  "
              f"T:{ties:>3} ({ties / total * 100:>5.1f}%)  "
              f"L:{losses:>3} ({losses / total * 100:>5.1f}%)")
    print()


if __name__ == "__main__":
    run_comparison_evaluation()
