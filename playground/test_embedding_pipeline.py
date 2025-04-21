from processors.dispatcher import ProcessorDispatcher
from embedding.factory import get_embedder
from embedding.processor import EmbeddingPipeline
from embedding.recursive_splitter import RecursiveSplitter
from database.qdrant_controller import QdrantVectorDB
from langchain.schema import Document

from typing import List
import uuid


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
    # 1. Load + chunk raw documents
    processor = ProcessorDispatcher(chunk_size=300, chunk_overlap=30)
    raw_docs = processor.process_directory("sandbox")

    # 2. Prepare embedder + splitter
    embedder = get_embedder("hf")  # or "titan"
    splitter = RecursiveSplitter(chunk_size=300, chunk_overlap=30)

    # 3. Run embedding pipeline
    pipeline = EmbeddingPipeline(embedder, splitter)
    texts, vectors = pipeline.run(raw_docs, use_batch=False)

    # 4. Wrap into Documents with metadata
    documents = prepare_documents(texts, raw_docs)

    # 5. Initialize and insert into Qdrant
    db = QdrantVectorDB()
    db.init_collection(vector_size=len(vectors[0]))
    db.insert_embeddings(documents=documents, vectors=vectors)

    print(f"Inserted {len(documents)} documents into Qdrant.")


if __name__ == "__main__":
    run_embedding_and_insert()
