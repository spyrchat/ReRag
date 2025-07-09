import os
from typing import Dict, Any
from database.qdrant_controller import QdrantVectorDB
from embedding.factory import get_embedder
from langchain_core.documents import Document
from logs.utils.logger import get_logger

logger = get_logger("retriever")


def retriever(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["question"]
    strategy = os.getenv("EMBEDDING_STRATEGY", "dense").lower()
    logger.info(f"[Retriever] Strategy: {strategy} | Query: {query}")

    try:
        db = QdrantVectorDB()
        dense_embedder = get_embedder(os.getenv("DENSE_EMBEDDER"))
        sparse_embedder = get_embedder(os.getenv("SPARSE_EMBEDDER"))

        vectorstore = db.as_langchain_vectorstore(
            dense_embedding=dense_embedder,
            sparse_embedding=sparse_embedder
        )

        docs: list[Document] = vectorstore.similarity_search(query, k=5)
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
