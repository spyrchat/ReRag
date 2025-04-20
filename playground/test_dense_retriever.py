from retrievers.dense_retriever import QdrantDenseRetriever
from database.qdrant_controller import QdrantVectorDB
from embedding.factory import get_embedder

if __name__ == "__main__":
    embedder = get_embedder(name="hf")  # or "hf", etc.

    db = QdrantVectorDB()
    client = db.get_client()
    collection_name = db.get_collection_name()

    retriever = QdrantDenseRetriever(
        client=client,
        collection_name=collection_name,
        embedding_model=embedder,
        top_k=5
    )

    query = "What is Hybrid RAG?"
    docs = retriever.get_relevant_documents(query)

    for i, doc in enumerate(docs):
        print(f"[{i+1}] Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}")
        print(doc.page_content[:200])
        print("-" * 40)
