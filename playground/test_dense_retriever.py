from retrievers.dense_retriever import QdrantDenseRetriever
from database.qdrant_controller import QdrantVectorDB
from embedding.factory import get_embedder

if __name__ == "__main__":
    # Load the embedder (e.g., "hf" for HuggingFace or "titan" for Bedrock Titan)
    embedder = get_embedder(name="hf")

    # Connect to Qdrant
    db = QdrantVectorDB()
    client = db.get_client()
    collection_name = db.get_collection_name()

    # Initialize the dense retriever
    retriever = QdrantDenseRetriever(
        client=client,
        collection_name=collection_name,
        embedding=embedder,
        top_k=5
    )

    # Perform retrieval
    query = "Advanced RAG Models with Graph Structures: Optimizing Complex Knowledge Reasoning and Text Generation"
    docs = retriever.get_relevant_documents(query)

    # Print results
    for i, doc in enumerate(docs):
        print(f"[{i+1}] Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}")
        print(f"Doc ID: {doc.metadata.get('doc_id', 'N/A')}")
        print("Text:")
        print(doc.page_content[:200])
        print("-" * 40)
