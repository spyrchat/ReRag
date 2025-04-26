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

    # 2. Load embedders
    dense_embedder = get_embedder(name=os.getenv("DENSE_EMBEDDER", "hf"))
    sparse_embedder = get_embedder(name=os.getenv(
        "SPARSE_EMBEDDER", "bm25"))  # <-- Load sparse too

    # 3. Create pipeline
    pipeline = EmbeddingPipeline(
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        splitter=splitter,
    )

    # 4. Run pipeline (produces all)
    texts, dense_vectors, sparse_vectors = pipeline.run(
        raw_docs, use_batch=False)

    # 5. Add metadata to chunks
    documents = prepare_documents(texts, raw_docs)

    # 6. Insert into Qdrant
    db = QdrantVectorDB()
    vector_size = len(dense_vectors[0]) if dense_vectors else 768  # fallback
    db.init_collection(vector_size=vector_size)

    db.insert_documents(
        documents=documents,
        dense_vectors=dense_vectors,
        sparse_vectors=sparse_vectors,
    )

    print(f"Inserted {len(documents)} documents into Qdrant.")


if __name__ == "__main__":
    run_embedding_and_insert()
