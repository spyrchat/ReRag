# playground/test_hybrid_retriever.py

import os
import dotenv

from database.qdrant_controller import QdrantVectorDB
from retrievers.hybrid_retriever import QdrantHybridRetriever
from embedding.factory import get_embedder

if __name__ == "__main__":
    # 1. Load .env and initialize embedders
    dotenv.load_dotenv(override=True)
    dense_embedder_name = os.getenv("DENSE_EMBEDDER", "hf")
    sparse_embedder_name = os.getenv("SPARSE_EMBEDDER", "bm25")

    dense_embedder = get_embedder(name=dense_embedder_name)
    sparse_embedder = get_embedder(name=sparse_embedder_name)
    print(f"Using dense embedder:  {dense_embedder}")
    print(f"Using sparse embedder: {sparse_embedder}")

    # 2. Initialize Qdrant database
    qdrant_db = QdrantVectorDB()
    qdrant_client = qdrant_db.client
    collection_name = qdrant_db.get_collection_name()

    # 3. Build the hybrid retriever
    hybrid_retriever = QdrantHybridRetriever(
        client=qdrant_client,
        collection_name=collection_name,
        dense_embedding=dense_embedder,
        sparse_embedding=sparse_embedder,
        top_k=5
    )

    # 4. Run a test query
    test_query = (
        "Advanced RAG Models with Graph Structures:"
        " Optimizing Complex Knowledge Reasoning and Text Generation"
    )
    results = hybrid_retriever.retrieve(test_query)

    if not results:
        print("No results found.")
    else:
        for i, (doc, score) in enumerate(results, start=1):
            metadata = doc.metadata or {}
            print(f"[{i}] score={score:.4f}")
            print(f"    source  : {metadata.get('source', 'N/A')}")
            print(f"    doc_id  : {metadata.get('doc_id', 'N/A')}")
            print(f"    chunk_id: {metadata.get('chunk_id', 'N/A')}")
            print("    excerpt :",
                  doc.page_content[:200].replace("\n", " "), "…")
            print("-" * 80)
