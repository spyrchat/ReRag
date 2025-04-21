from retrievers.dense_retriever import QdrantDenseRetriever
from database.qdrant_controller import QdrantVectorDB
from embedding.factory import get_embedder
from langchain_core.documents import Document
import os
import dotenv

# Load environment variables
dotenv.load_dotenv(override=True)

if __name__ == "__main__":
    # Load embedder from env
    embedder = get_embedder(name=os.getenv("DENSE_EMBEDDER"))
    print(f"Using embedder: {embedder}")
    # Initialize DB and LangChain-compatible vectorstore
    db = QdrantVectorDB()
    vectorstore = db.as_langchain_vectorstore(embedding=embedder)

    # Create retriever
    retriever = QdrantDenseRetriever(
        embedding=embedder,
        vectorstore=vectorstore,
        top_k=5
    )

    # Run query
    query = "Advanced RAG Models with Graph Structures: Optimizing Complex Knowledge Reasoning and Text Generation"
    results = retriever.get_relevant_documents(query)

    if not results:
        print("No results found.")
    else:
        for i, (doc, score) in enumerate(results):
            print(f"[{i+1}] Score: {score:.4f}")
            print(
                f"Doc ID: {doc.metadata.get('doc_id', 'N/A')} | Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}")
            print("Text:")
            print(doc.page_content[:200])
            print("-" * 50)
