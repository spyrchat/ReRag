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


def run_embedding_and_insert():
    dotenv.load_dotenv()

    # Load and chunk
    processor = ProcessorDispatcher(chunk_size=300, chunk_overlap=30)
    raw_docs = processor.process_directory("sandbox")

    splitter = RecursiveSplitter(chunk_size=300, chunk_overlap=30)
    embedder = get_embedder(os.getenv("DENSE_EMBEDDER", "hf"))

    chunks = splitter.split(raw_docs)

    # Add metadata
    documents = prepare_documents(
        [doc.page_content for doc in chunks], raw_docs)

    # Insert via LangChain
    db = QdrantVectorDB()
    db.init_collection(vector_size=embedder.embed_query("test").__len__())
    db.insert_documents(documents=documents, embedding=embedder)

    print(f"✅ Inserted {len(documents)} documents into Qdrant.")


if __name__ == "__main__":
    run_embedding_and_insert()
