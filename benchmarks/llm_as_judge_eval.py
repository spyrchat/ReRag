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
# change per provider ex "gpt-4.1" for OpenAI, "claude-sonnet-4-0" for Anthropic
MODEL_NAME = "gpt-5"  # gpt-4.1 | claude-sonnet-4-0

INPUT_PATH = "/home/spiros/Desktop/Thesis/results/ground_truth_for_test/ground_truth_intermediate.json"
OUTPUT_PATH = f"/home/spiros/Desktop/Thesis/results/llm_judge_scores/llm_judge_scores_{PROVIDER}_{MODEL_NAME.replace('.', '-')}.jsonl"
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
You are a rigorous, impartial evaluator. Your job is to assess an assistant's answer to a user's energy-related question using structured context data.

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

You must respond in this **strict JSON** format (no code blocks, no commentary):

{{
  "faithfulness": int,      // 1 to 5
  "relevance": int,         // 1 to 5
  "helpfulness": int,       // 1 to 5
  "justification": "Concise explanation referencing specific parts of the answer and context"
}}

### Evaluation Example

User Question:
{entry.get("question", "No question provided.")}

Assistant's Answer:
{entry.get("answer", "No answer provided.")}

Context:
{context}
""".strip()


def run_llm_judge_evaluation() -> pd.Series:
    """
    Runs LLM-based evaluation for each entry in the input file.

    Returns:
        pd.Series: Mean scores for faithfulness, relevance, helpfulness.
    """
    with open(INPUT_PATH, "r", encoding="utf-8") as infile:
        entries = json.load(infile)  # Load JSON array directly

    results = []
    for i in tqdm(range(0, len(entries), BATCH_SIZE), desc=f"Evaluating with {PROVIDER}/{MODEL_NAME}"):
        batch = entries[i:i + BATCH_SIZE]
        prompts = [build_prompt(entry) for entry in batch]

        for prompt, entry in zip(prompts, batch):
            try:
                score = judge.evaluate(prompt)
                results.append({
                    "question": entry.get("question", ""),
                    "faithfulness": score.get("faithfulness", 0),
                    "relevance": score.get("relevance", 0),
                    "helpfulness": score.get("helpfulness", 0),
                    "justification": score.get("justification", ""),
                    "answer": entry.get("answer", ""),
                    "has_context": bool(entry.get("context"))
                })
            except Exception as e:
                logging.error(
                    f"Evaluation failed for question: {entry.get('question', '')}")
                logging.error(f"Error: {e}")
                results.append({
                    "question": entry.get("question", ""),
                    "faithfulness": 0,
                    "relevance": 0,
                    "helpfulness": 0,
                    "justification": f"Evaluation failed: {str(e)}",
                    "answer": entry.get("answer", ""),
                    "has_context": bool(entry.get("context"))
                })

        time.sleep(SLEEP_BETWEEN_BATCHES)

    # Save results
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in results:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    summary = run_llm_judge_evaluation()
    print("\n=== LLM-as-a-Judge Evaluation Complete ===")
    print(summary)
