from typing import Dict, Any, List
from langchain_core.documents import Document
from logs.utils.logger import get_logger
from bin.agent_retriever import ConfigurableRetrieverAgent

logger = get_logger(__name__)


def make_configurable_retriever(config_path: str = None, cache_pipeline: bool = True):
    """
    Factory to return a configurable retriever node.

    Args:
        config_path (str, optional): Path to YAML configuration file for retrieval pipeline.
                                   If None, will load from main config.yml
        cache_pipeline (bool): Whether to cache the pipeline for reuse

    Returns:
        function: Retriever node function that can be used in LangGraph agent
    """
    # Load config path from main config if not provided
    if config_path is None:
        from config.config_loader import load_config
        main_config = load_config()
        config_path = main_config.get("agent_retrieval", {}).get("config_path",
                                                                 "pipelines/configs/retrieval/modern_hybrid.yml")

    # Initialize the configurable retriever agent
    agent = ConfigurableRetrieverAgent(config_path, cache_pipeline)

    # Log configuration info
    config_info = agent.get_config_info()
    logger.info(f"[Retriever] Initialized with config: {config_path}")
    logger.info(
        f"[Retriever] Pipeline: {config_info['retriever_type']} with {config_info['num_stages']} stages")
    logger.info(
        f"[Retriever] Components: {', '.join(config_info['stage_types'])}")

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

        try:
            # Get retrieval configuration
            top_k = state.get("retrieval_top_k",
                              config_info.get("retriever_top_k", 5))

            # Retrieve documents using configurable pipeline
            docs_info = agent.retrieve(query, top_k=top_k)

            # Convert to context string and preserve metadata
            context_parts = []
            retrieved_docs = []

            for doc_info in docs_info:
                # Add to context
                content = doc_info["content"]
                context_parts.append(content)

                # Create Document object with metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        "score": doc_info["score"],
                        "retrieval_method": doc_info["retrieval_method"],
                        "question_title": doc_info["question_title"],
                        "tags": doc_info["tags"],
                        "external_id": doc_info["external_id"],
                        "enhanced": doc_info["enhanced"],
                        "answer_quality": doc_info["answer_quality"]
                    }
                )
                retrieved_docs.append(doc)

            context = "\n\n".join(context_parts)

            logger.info(
                f"[Retriever] Retrieved {len(docs_info)} documents using {config_info['retriever_type']}")
            logger.info(
                f"[Retriever] Pipeline components: {', '.join(config_info['stage_types'])}")

            # Return enhanced state with retrieval metadata
            return {
                **state,
                "context": context,
                "retrieved_documents": retrieved_docs,
                "retrieval_metadata": {
                    "num_results": len(docs_info),
                    "retrieval_method": docs_info[0]["retrieval_method"] if docs_info else "none",
                    "pipeline_config": config_info,
                    "top_result_score": docs_info[0]["score"] if docs_info else 0.0
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

    return retriever


def make_retriever(db, dense_embedder, sparse_embedder, top_k=5, strategy=None):
    """
    Legacy retriever factory for backward compatibility.
    Consider migrating to make_configurable_retriever for better flexibility.
    """
    def retriever(state: Dict[str, Any]) -> Dict[str, Any]:
        query = state["question"]
        logger.info(f"[Retriever] Query: {query}")
        if strategy:
            logger.info(f"[Retriever] Retrieval strategy: {strategy}")

        try:
            vectorstore = db.as_langchain_vectorstore(
                dense_embedding=dense_embedder,
                sparse_embedding=sparse_embedder,
            )

            docs: List[Document] = vectorstore.similarity_search(
                query, k=top_k)
            context = "\n\n".join([doc.page_content for doc in docs])

            logger.info(f"[Retriever] Retrieved {len(docs)} documents.")
            return {
                **state,
                "context": context
            }

        except Exception as e:
            logger.error(f"[Retriever] Retrieval failed: {str(e)}")
            return {
                **state,
                "context": "",
                "error": f"Retriever failed: {str(e)}"
            }
    return retriever
