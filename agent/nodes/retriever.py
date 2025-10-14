from typing import Dict, Any, List
from langchain_core.documents import Document
from logs.utils.logger import get_logger
from components.retrieval_pipeline import RetrievalPipelineFactory
from config.config_loader import load_config

logger = get_logger(__name__)

# Load retrieval pipeline once at module level
main_config = load_config()
retrieval_config_path = main_config.get("agent_retrieval", {}).get(
    "config_path", "pipelines/configs/retrieval/fast_dense_bge_m3.yml")
retrieval_config = load_config(retrieval_config_path)
pipeline = RetrievalPipelineFactory.create_from_config(retrieval_config)

logger.info(
    f"[Retriever] Initialized pipeline with config: {retrieval_config_path}")
logger.info(
    f"[Retriever] Pipeline components: {[c.component_name for c in pipeline.components]}")


def _format_docs_for_agent(results) -> tuple[str, List[Document]]:
    """
    Simple formatting function to convert retrieval results for agent state.

    Args:
        results: List of RetrievalResult objects from pipeline

    Returns:
        tuple: (context_string, list_of_documents)
    """
    context_parts = []
    retrieved_docs = []

    for result in results:
        content = result.document.page_content
        context_parts.append(content)

        # Create Document object with metadata
        doc = Document(
            page_content=content,
            metadata={
                "score": result.score,
                "retrieval_method": result.retrieval_method,
                **result.document.metadata
            }
        )
        retrieved_docs.append(doc)

    context = "\n\n".join(context_parts)
    return context, retrieved_docs


def retriever(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retriever node function for LangGraph agent.

    Args:
        state (Dict[str, Any]): Current agent state containing question and other context

    Returns:
        Dict[str, Any]: Updated state with retrieved documents and metadata
    """
    query = state["question"]
    logger.info(f"[Retriever] Query: {query}")

    # Skip retrieval if context already exists (e.g., from test fixtures)
    if "context" in state and state["context"]:
        logger.info(f"[Retriever] Context already exists ({len(state['context'])} chars), skipping retrieval")
        return state

    try:
        # Get top_k from state or use default
        top_k = state.get("top_k", 10)

        # Run retrieval directly through pipeline
        results = pipeline.run(query, k=top_k)

        # Format results for agent state
        context, retrieved_docs = _format_docs_for_agent(results)

        logger.info(f"[Retriever] Retrieved {len(results)} documents")
        logger.info(f"[Retriever] Context length: {len(context)} characters")

        if not context or not context.strip():
            logger.warning(
                "[Retriever] WARNING: Context is empty or whitespace only!")

        # Return enhanced state with retrieval metadata
        return {
            **state,
            "context": context,
            "retrieved_documents": retrieved_docs,
            "retrieval_metadata": {
                "num_results": len(results),
                "retrieval_method": results[0].retrieval_method if results else "none",
                "top_result_score": results[0].score if results else 0.0
            }
        }

    except Exception as e:
        logger.error(f"[Retriever] Retrieval failed: {str(e)}")
        return {
            **state,
            "context": "",
            "error": f"Retriever failed: {str(e)}",
            "retrieval_metadata": {
                "num_results": 0,
                "error": str(e)
            }
        }
