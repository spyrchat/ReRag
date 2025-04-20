from retrievers.dense_retriever import QdrantDenseRetriever
from database.qdrant_controller import QdrantVectorDB
from embedding.factory import get_embedder
import logging
import dotenv
import os
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Step 1: Load embedder
    embedder = get_embedder(name="hf")

    # Step 2: Qdrant setup
    db = QdrantVectorDB()
    client = db.get_client()
    collection_name = db.get_collection_name()

    # Step 3: Create retriever
    retriever = QdrantDenseRetriever(
        client=client,
        collection_name=collection_name,
        embedding_model=embedder,
        top_k=5
    )

    # Step 4: Query
    query = "What is Hybrid RAG?"
    results = retriever.get_relevant_documents(query)

    # Step 5: Display results
    print(f"\nQuery: {query}")
    print("-" * 50)
    for i, doc in enumerate(results):
        print(f"[{i+1}] Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}")
        print(doc.page_content[:300])
        print("-" * 50)
