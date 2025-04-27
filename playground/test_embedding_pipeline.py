from processors.dispatcher import ProcessorDispatcher
from embedding.factory import get_embedder
from embedding.processor import EmbeddingPipeline
from embedding.recursive_splitter import RecursiveSplitter
from database.qdrant_controller import QdrantVectorDB
from langchain.schema import Document
import os
import dotenv
import uuid
from typing import List

# Function to prepare documents with metadata


def prepare_documents(texts: List[str], original_docs: List[Document]) -> List[Document]:
    enriched = []
    for i, text in enumerate(texts):
        source_doc = original_docs[i % len(original_docs)]
        enriched.append(
            Document(
                page_content=text,
                metadata={
                    "source": source_doc.metadata.get("source", "unknown"),
                    "doc_id": source_doc.metadata.get("doc_id", str(uuid.uuid4())),
                    "chunk_id": i
                }
            )
        )
    return enriched


def run_embedding_and_insert():
    dotenv.load_dotenv(override=True)

    # 1. Load and chunk
    processor = ProcessorDispatcher(chunk_size=300, chunk_overlap=30)
    raw_docs = processor.process_directory("sandbox")

    splitter = RecursiveSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split(raw_docs)

    # 2. Prepare Documents
    documents = prepare_documents(
        texts=[doc.page_content for doc in chunks],
        original_docs=raw_docs
    )

    # 3. Load Embedders
    dense_embedder = get_embedder(name=os.getenv("DENSE_EMBEDDER", "hf"))
    sparse_embedder = get_embedder(name=os.getenv("SPARSE_EMBEDDER", "bm25"))

    # 4. Insert into Qdrant
    db = QdrantVectorDB()
    db.insert_documents(
        documents=documents,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
    )

    print(f"Inserted {len(documents)} documents into Qdrant.")


if __name__ == "__main__":
    run_embedding_and_insert()
