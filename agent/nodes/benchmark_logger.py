"""
Benchmark Logger Node - Saves pipeline execution data for benchmarking.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any
from pathlib import Path
from logs.utils.logger import get_logger

logger = get_logger(__name__)


class BenchmarkLogger:
    """Handles logging of agent pipeline executions for benchmarking."""

    def __init__(self, output_dir: str = "logs/benchmark", enabled: bool = True):
        """
        Initialize benchmark logger.

        Args:
            output_dir: Directory to save benchmark logs
            enabled: Whether logging is enabled
        """
        self.output_dir = Path(output_dir)
        self.enabled = enabled

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"[BenchmarkLogger] Initialized (output_dir={output_dir})")
        else:
            logger.info("[BenchmarkLogger] Disabled")

    def log_execution(self, state: Dict[str, Any]) -> None:
        """
        Log a single pipeline execution.

        Args:
            state: Final agent state containing all pipeline data
        """
        if not self.enabled:
            return

        try:
            # Extract only what's needed for benchmarking
            execution_data = {
                "user_question": state.get("question", ""),
                "llm_answer": state.get("answer", ""),
                "context_provided": state.get("context", "")
            }

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"execution_{timestamp}.json"
            filepath = self.output_dir / filename

            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(execution_data, f, ensure_ascii=False, indent=2)

            logger.info(f"[BenchmarkLogger] Saved execution to {filename}")

        except Exception as e:
            logger.error(
                f"[BenchmarkLogger] Failed to log execution: {str(e)}")

    def _serialize_documents(self, documents: list) -> list:
        """
        Serialize LangChain documents to JSON-compatible format.

        Args:
            documents: List of Document objects

        Returns:
            List of dictionaries
        """
        serialized = []
        for doc in documents:
            try:
                doc_dict = {
                    "content": doc.page_content if hasattr(doc, 'page_content') else str(doc),
                    "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
                }
                serialized.append(doc_dict)
            except Exception as e:
                logger.warning(
                    f"[BenchmarkLogger] Failed to serialize document: {str(e)}")
                serialized.append({"content": str(doc), "metadata": {}})

        return serialized

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of logged executions.

        Returns:
            Dictionary with summary statistics
        """
        if not self.enabled:
            return {"enabled": False}

        try:
            log_files = list(self.output_dir.glob("execution_*.json"))

            total_executions = len(log_files)
            retrieval_count = 0
            direct_count = 0
            error_count = 0

            for log_file in log_files:
                try:
                    with open(log_file, 'r') as f:
                        data = json.load(f)
                        if data.get("needs_retrieval"):
                            retrieval_count += 1
                        else:
                            direct_count += 1
                        if data.get("error"):
                            error_count += 1
                except:
                    pass

            return {
                "enabled": True,
                "output_dir": str(self.output_dir),
                "total_executions": total_executions,
                "with_retrieval": retrieval_count,
                "direct_answer": direct_count,
                "errors": error_count
            }
        except Exception as e:
            logger.error(f"[BenchmarkLogger] Failed to get summary: {str(e)}")
            return {"enabled": True, "error": str(e)}


# Global benchmark logger instance
_benchmark_logger = None


def initialize_benchmark_logger(output_dir: str = "logs/benchmark", enabled: bool = True):
    """
    Initialize the global benchmark logger.

    Args:
        output_dir: Directory to save benchmark logs
        enabled: Whether logging is enabled
    """
    global _benchmark_logger
    _benchmark_logger = BenchmarkLogger(output_dir, enabled)
    return _benchmark_logger


def get_benchmark_logger() -> BenchmarkLogger:
    """Get the global benchmark logger instance."""
    global _benchmark_logger
    if _benchmark_logger is None:
        _benchmark_logger = BenchmarkLogger()
    return _benchmark_logger


def make_benchmark_logger_node():
    """
    Factory to create a benchmark logger node.

    Returns:
        function: Benchmark logger node function
    """
    bench_logger = get_benchmark_logger()

    def benchmark_logger_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log pipeline execution for benchmarking.

        Args:
            state: Current agent state

        Returns:
            Unchanged state (pass-through node)
        """
        logger.info("[BenchmarkLogger] Logging execution data")
        bench_logger.log_execution(state)
        return state

    return benchmark_logger_node
