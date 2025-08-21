import os
import uuid
import logging
from typing import List
import dotenv

from langchain.schema import Document
from embedding.factory import get_embedder
from embedding.recursive_splitter import RecursiveSplitter
from database.qdrant_controller import QdrantVectorDB

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def prepare_documents(texts: List[str], original_docs: List[Document]) -> List[Document]:
    enriched = []
    for i, text in enumerate(texts):
        src = original_docs[i % len(original_docs)]
        enriched.append(
            Document(
                page_content=text,
                metadata={
                    "source": src.metadata.get("source", "unknown"),
                    "doc_id": src.metadata.get("doc_id", str(uuid.uuid4())),
                    "chunk_id": i
                }
            )
        )
    return enriched


def run_embedding_and_insert():
    dotenv.load_dotenv(override=True)

    # 1. Load + chunk
    processor = ProcessorDispatcher(chunk_size=300, chunk_overlap=30)
    raw_docs = processor.process_directory("sandbox")
    splitter = RecursiveSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split(raw_docs)
    documents = prepare_documents(
        texts=[c.page_content for c in chunks],
        original_docs=raw_docs
    )

    # 2. embedders
    dense_embedder = get_embedder(os.getenv("DENSE_EMBEDDER", "hf"))
    sparse_embedder = get_embedder(os.getenv("SPARSE_EMBEDDER", "bm25"))

    # 3. init Qdrant
    db = QdrantVectorDB()

    # compute your dense dimension once:
    dq = dense_embedder.embed_query("test")
    if hasattr(dq, "shape"):
        dense_dim = dq.shape[-1]
    else:
        dense_dim = len(dq)

    db.init_collection(dense_vector_size=dense_dim)

    # 4. insert
    db.insert_documents(
        documents=documents,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder
    )
    print(f"Inserted {len(documents)} documents into Qdrant.")


if __name__ == "__main__":
    run_embedding_and_insert()
