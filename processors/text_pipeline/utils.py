from langchain.schema import Document
from typing import List
from processors.core.metadata import PageMetadata
import uuid


def prepare_documents(texts: List[str], original_docs: List[Document]) -> List[Document]:
    enriched = []
    for i, text in enumerate(texts):
        src = original_docs[i % len(original_docs)]

        metadata = PageMetadata(
            doc_id=src.metadata.get("doc_id", str(uuid.uuid4())),
            page=src.metadata.get("page", 0),
            source=src.metadata.get("source", "unknown"),
        )

        enriched.append(Document(
            page_content=text,
            metadata={**metadata.to_dict(), "chunk_id": i}
        ))
    return enriched
