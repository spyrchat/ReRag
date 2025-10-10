"""
LLM-as-a-Judge Evaluation Script (Multi-Provider)

This script performs batch evaluation of RAG-generated answers using a structured LLM prompt
and scores each example on:
- Faithfulness
- Relevance
- Helpfulness

It supports pluggable backends (OpenAI, Anthropic, Cohere, etc.) to reduce model bias.

Expected Input (`benchmark_dataset_preprocessed.jsonl`):
- question
- answer
- contexts (list or string)
- ground_truth

Output:
- JSONL with evaluation scores and justification
- Printed summary of average scores
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd
# Make sure this path matches your project
from llm_judge import LLMJudge

# === Config ===
PROVIDER = "openai"       # openai | anthropic
# Valid OpenAI models: gpt-4o, gpt-4o-mini, gpt-4-turbo
# Valid Anthropic models: claude-3-5-sonnet-20241022, claude-3-opus-20240229
MODEL_NAME = "gpt-5"

INPUT_PATH = "/home/spiros/Desktop/Thesis/results/test_standard/ground_truth_self_rag_20251009_221311.json"
OUTPUT_PATH = f"/home/spiros/Desktop/Thesis/results/llm_judge_scores/llm_judge_scores_self_rag_{PROVIDER}_{MODEL_NAME.replace('.', '-')}.jsonl"
BATCH_SIZE = 20
SLEEP_BETWEEN_BATCHES = 0

# === Setup Logging ===
import logging

logger = logging.getLogger(__name__)

# === Load environment ===
load_dotenv()

# === Initialize LLM Judge ===
judge = LLMJudge(provider=PROVIDER, model_name=MODEL_NAME)


def build_prompt(entry: Dict[str, str]) -> str:
    """
    Builds a G-Eval-style prompt for evaluating a RAG answer with faithfulness, relevance, and helpfulness scores.

    Args:
        entry (dict): Contains 'question', 'contexts', 'ground_truth', 'answer'.

    Returns:
        str: A structured prompt string for LLM-based judgment.
    """
    # Safely extract context
    raw_context = entry.get("context", "")
    if isinstance(raw_context, list):
        context = raw_context[0] if raw_context else "No context provided."
    elif isinstance(raw_context, str):
        context = raw_context
    else:
        context = str(raw_context) if raw_context else "No context provided."

    return f"""
You are a rigorous, impartial evaluator. Your job is to assess an assistant's answer to a user's software engineering question using structured context data.

For each evaluation dimension, assign an integer score from 1 (poor) to 5 (excellent), using the detailed rubric below.
Justify your ratings with specific evidence from the answer and context.

### Dimensions

1. Faithfulness (Groundedness):
- 5: Fully accurate and directly supported by the context.
- 4: Mostly accurate with minor unsupported details.
- 3: Somewhat supported but contains vague or partially incorrect claims.
- 2: Mostly unsupported or contains significant inaccuracies.
- 1: Fabricated or contradicts the context.

2. Relevance (Answering the User Question):
- 5: Directly and completely answers all parts of the user’s question.
- 4: Addresses most aspects clearly, minor omissions allowed.
- 3: Partially answers the question, or some irrelevant info.
- 2: Barely related to the actual question.
- 1: Completely off-topic.

3. Helpfulness (Clarity and Usefulness):
- 5: Very informative, easy to follow, and actionable.
- 4: Generally helpful with minor clarity or completeness issues.
- 3: Understandable but limited in depth or insight.
- 2: Hard to follow or missing useful details.
- 1: Confusing or unhelpful.

### Format

You must respond in this **strict JSON** format (no code blocks, no commentary).
IMPORTANT: Properly escape any quotes, newlines, or special characters in the justification string.

{{
  "faithfulness": 4,
  "relevance": 5,
  "helpfulness": 4,
  "justification": "Brief explanation without special characters that break JSON"
}}

Return ONLY valid JSON, nothing else.

### Evaluation

User Question:
{entry.get("question", "No question provided.")}

Assistant's Answer:
{entry.get("answer", "No answer provided.")}

Context:
{context}
""".strip()


def run_llm_judge_evaluation() -> Dict[str, float]:
    """
    Runs LLM-based evaluation for each entry in the input file.

    Returns:
        Dict[str, float]: Mean scores for faithfulness, relevance, helpfulness.
    """
    # Ensure output directory exists
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {INPUT_PATH}")
    with open(INPUT_PATH, "r", encoding="utf-8") as infile:
        entries = json.load(infile)  # Load JSON array directly

    print(f"Loaded {len(entries)} entries")
    print(f"Output will be saved to: {OUTPUT_PATH}")

    results = []
    for i in tqdm(range(0, len(entries), BATCH_SIZE), desc=f"Evaluating with {PROVIDER}/{MODEL_NAME}"):
        batch = entries[i:i + BATCH_SIZE]
        prompts = [build_prompt(entry) for entry in batch]

        for prompt, entry in zip(prompts, batch):
            max_retries = 3
            retry_count = 0
            success = False

            while retry_count < max_retries and not success:
                try:
                    score = judge.evaluate(prompt)

                    # Validate the response has required fields
                    if not all(k in score for k in ["faithfulness", "relevance", "helpfulness"]):
                        raise ValueError(
                            f"Missing required fields in response: {score}")

                    # Validate score ranges
                    for key in ["faithfulness", "relevance", "helpfulness"]:
                        if not (1 <= score[key] <= 5):
                            raise ValueError(
                                f"{key} score {score[key]} out of range [1-5]")

                    results.append({
                        "question": entry.get("question", ""),
                        "faithfulness": score.get("faithfulness", 0),
                        "relevance": score.get("relevance", 0),
                        "helpfulness": score.get("helpfulness", 0),
                        "justification": score.get("justification", ""),
                        "answer": entry.get("answer", ""),
                        "has_context": bool(entry.get("context"))
                    })
                    success = True

                except json.JSONDecodeError as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        logging.warning(
                            f"JSON parsing failed (attempt {retry_count}/{max_retries}), retrying...")
                        time.sleep(1)  # Brief delay before retry
                    else:
                        logging.error(
                            f"Evaluation failed after {max_retries} attempts for question: {entry.get('question', '')[:100]}...")
                        logging.error(f"Error: {e}")
                        results.append({
                            "question": entry.get("question", ""),
                            "faithfulness": 0,
                            "relevance": 0,
                            "helpfulness": 0,
                            "justification": f"Evaluation failed after {max_retries} retries: JSON parsing error",
                            "answer": entry.get("answer", ""),
                            "has_context": bool(entry.get("context"))
                        })

                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        logging.warning(
                            f"Evaluation failed (attempt {retry_count}/{max_retries}): {str(e)[:100]}, retrying...")
                        time.sleep(1)
                    else:
                        logging.error(
                            f"Evaluation failed after {max_retries} attempts for question: {entry.get('question', '')[:100]}...")
                        logging.error(f"Error: {e}")
                        results.append({
                            "question": entry.get("question", ""),
                            "faithfulness": 0,
                            "relevance": 0,
                            "helpfulness": 0,
                            "justification": f"Evaluation failed after {max_retries} retries: {str(e)[:100]}",
                            "answer": entry.get("answer", ""),
                            "has_context": bool(entry.get("context"))
                        })

        time.sleep(SLEEP_BETWEEN_BATCHES)

    # Save results
    print(f"\nSaving {len(results)} results to: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in results:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ Results saved successfully!")

    # Calculate summary statistics
    valid_results = [r for r in results if r['faithfulness'] > 0]

    if not valid_results:
        print("⚠️  No valid results to summarize")
        return {}

    summary = {
        "total_evaluated": len(results),
        "successful": len(valid_results),
        "failed": len(results) - len(valid_results),
        "mean_faithfulness": sum(r['faithfulness'] for r in valid_results) / len(valid_results),
        "mean_relevance": sum(r['relevance'] for r in valid_results) / len(valid_results),
        "mean_helpfulness": sum(r['helpfulness'] for r in valid_results) / len(valid_results)
    }

    return summary


if __name__ == "__main__":
    summary = run_llm_judge_evaluation()
    print("\n=== LLM-as-a-Judge Evaluation Complete ===")
    print(f"Total evaluated: {summary.get('total_evaluated', 0)}")
    print(f"Successful: {summary.get('successful', 0)}")
    print(f"Failed: {summary.get('failed', 0)}")
    print(f"\nMean Scores:")
    print(f"  Faithfulness: {summary.get('mean_faithfulness', 0):.3f}")
    print(f"  Relevance:    {summary.get('mean_relevance', 0):.3f}")
    print(f"  Helpfulness:  {summary.get('mean_helpfulness', 0):.3f}")
