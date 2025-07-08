import os
from retrievers.dense_retriever import QdrantDenseRetriever
from retrievers.hybrid_retriever import QdrantHybridRetriever
from retrievers.sql_retriever import SQLRetriever
from database.qdrant_controller import QdrantVectorDB
from embedding.factory import get_embedder


class RetrieverRouter:
    def __init__(self):
        self.strategy = os.getenv("RETRIEVER_STRATEGY", "auto").lower()

        self.qdrant = QdrantVectorDB()
        self.dense_model = get_embedder(name=os.getenv("DENSE_EMBEDDER"))
        self.sparse_model = get_embedder(name=os.getenv("SPARSE_EMBEDDER"))

        if self.strategy == "dense":
            self.retriever = QdrantDenseRetriever(
                embedding=self.dense_model,
                vectorstore=self.qdrant.as_langchain_vectorstore(
                    dense_embedding=self.dense_model)
            )
        elif self.strategy == "hybrid":
            self.retriever = QdrantHybridRetriever(
                client=self.qdrant.get_client(),
                collection_name=self.qdrant.get_collection_name(),
                dense_embedding=self.dense_model,
                sparse_embedding=self.sparse_model,
                dense_vector_name=os.getenv("DENSE_VECTOR_NAME", "dense"),
                sparse_vector_name=os.getenv("SPARSE_VECTOR_NAME", "sparse")
            )
        elif self.strategy == "sql":
            self.retriever = SQLRetriever()
        else:
            # Auto: pick dynamically at query time
            self.dense_retriever = QdrantDenseRetriever(
                embedding=self.dense_model,
                vectorstore=self.qdrant.as_langchain_vectorstore(
                    dense_embedding=self.dense_model)
            )
            self.hybrid_retriever = QdrantHybridRetriever(
                client=self.qdrant.get_client(),
                collection_name=self.qdrant.get_collection_name(),
                dense_embedding=self.dense_model,
                sparse_embedding=self.sparse_model,
                dense_vector_name=os.getenv("DENSE_VECTOR_NAME", "dense"),
                sparse_vector_name=os.getenv("SPARSE_VECTOR_NAME", "sparse")
            )
            self.sql_retriever = SQLRetriever()

    def retrieve(self, query: str):
        if self.strategy in ("dense", "hybrid", "sql"):
            return self.retriever.retrieve(query)

        # Auto strategy logic
        if self._is_sql_query(query):
            return self.sql_retriever.retrieve(query)
        if self._is_table_like(query):
            return self.hybrid_retriever.retrieve(query)
        return self.dense_retriever.retrieve(query)

    def _is_sql_query(self, query: str) -> bool:
        return any(kw in query.lower() for kw in ["select", "from", "where", "table"])

    def _is_table_like(self, query: str) -> bool:
        return any(kw in query.lower() for kw in ["how many", "total", "sum", "average", "percentage"])
