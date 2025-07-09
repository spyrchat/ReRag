from typing import Dict, Any, List
from langchain_core.documents import Document
from logs.utils.logger import get_logger

logger = get_logger(__name__)


def make_retriever(db, dense_embedder, sparse_embedder, top_k=5, strategy=None):
    """
    Factory to return a retriever node with pre-initialized dependencies.
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
