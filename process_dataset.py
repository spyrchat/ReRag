from beir.datasets.data_loader import GenericDataLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo.collection import Collection
from typing import List
from mongodb_utils import connect_to_mongodb
from embeddings import TitanEmbeddingWrapper
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("process_dataset.log", mode="w")
    ]
)

# Global variable to set the database and collection names
DATABASE_NAME = "aws_gen_ai"
COLLECTION_NAME = "TrecCovid"

# Initialize the TitanEmbeddingWrapper globally
embedding_wrapper = TitanEmbeddingWrapper(model="amazon.titan-embed-text-v2:0")


class BaseChunker:
    """
    Abstract base class for chunking strategies.
    """

    def split_documents(self, documents: List[Document]) -> List[Document]:
        raise NotImplementedError(
            "This method should be implemented by subclasses.")


class SemanticChunkerWrapper(BaseChunker):
    """
    Wrapper for SemanticChunker.
    """

    def __init__(self, embedding_wrapper):
        self.splitter = SemanticChunker(
            embeddings=embedding_wrapper,
            breakpoint_threshold_type="gradient",
            breakpoint_threshold_amount=0.8,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        logging.info("Using SemanticChunker to split documents.")
        return self.splitter.split_documents(documents)


class TextSplitterWrapper(BaseChunker):
    """
    Wrapper for LangChain's TextSplitter.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        logging.info("Using TextSplitter to split documents.")
        chunked_docs = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc.page_content)
            for chunk in chunks:
                chunked_docs.append(
                    Document(page_content=chunk, metadata=doc.metadata))
        return chunked_docs


class DocumentProcessor:
    """
    Class to process and store documents.
    """

    def __init__(self, chunker: BaseChunker, embedding_wrapper, collection: Collection):
        self.chunker = chunker
        self.embedding_wrapper = embedding_wrapper
        self.collection = collection

    def process_documents(self, documents: List[Document], batch_size: int = 250):
        logging.info(
            f"Starting batch processing with batch size: {batch_size}")

        for batch_start in range(0, len(documents), batch_size):
            batch = documents[batch_start:batch_start + batch_size]
            logging.info(
                f"Processing batch {batch_start // batch_size + 1} with {len(batch)} documents.")

            # Split documents into chunks
            chunked_docs = self.chunker.split_documents(batch)
            logging.info(
                f"Batch {batch_start // batch_size + 1}: Split {len(batch)} documents into {len(chunked_docs)} chunks.")

            # Generate embeddings for each chunk
            texts = [chunk.page_content for chunk in chunked_docs]
            logging.info(
                f"Batch {batch_start // batch_size + 1}: Starting embedding generation for {len(texts)} chunks...")
            try:
                embeddings = self.embedding_wrapper.embed_documents(texts)
                logging.info(
                    f"Batch {batch_start // batch_size + 1}: Successfully generated embeddings for {len(texts)} chunks.")
            except Exception as e:
                logging.error(
                    f"Batch {batch_start // batch_size + 1}: Failed to generate embeddings. Error: {e}")
                continue  # Skip this batch and move to the next

            # Prepare the chunks for insertion
            to_insert = []
            for i, chunk in enumerate(chunked_docs):
                text = chunk.page_content
                metadata = chunk.metadata

                to_insert.append({
                    "doc_id": metadata.get("doc_id"),
                    "chunk_id": f"{metadata.get('doc_id')}_{i}",
                    "text": text,
                    "embedding": embeddings[i],
                    "metadata": metadata,
                    "source": "trec-covid"
                })

            # Insert the processed chunks into the MongoDB collection
            if to_insert:
                try:
                    logging.info(
                        f"Batch {batch_start // batch_size + 1}: Inserting {len(to_insert)} chunks into MongoDB...")
                    self.collection.insert_many(to_insert)
                    logging.info(
                        f"Batch {batch_start // batch_size + 1}: Successfully inserted {len(to_insert)} chunks into the collection '{self.collection.name}'.")
                except Exception as e:
                    logging.error(
                        f"Batch {batch_start // batch_size + 1}: Failed to insert chunks into MongoDB. Error: {e}")

        logging.info("Batch processing completed successfully.")


def load_all_documents(corpus_path: str) -> List[Document]:
    """
    Load all documents from the specified corpus path.

    Args:
        corpus_path (str): Path to the corpus folder.

    Returns:
        List[Document]: List of LangChain Document objects.
    """
    logging.info(f"Loading all documents from the dataset at {corpus_path}.")
    corpus, _, _ = GenericDataLoader(
        data_folder=corpus_path).load(split="test")
    documents = []

    for doc_id, content in corpus.items():
        title = content.get("title", "")
        text = content.get("text", "")
        full_text = f"{title}. {text}".strip()
        if not full_text:
            logging.warning(
                f"Document {doc_id} has no content and will be skipped.")
            continue
        documents.append(Document(page_content=full_text, metadata={
            "doc_id": doc_id,
            "title": title
        }))

    logging.info(f"Loaded {len(documents)} documents from the dataset.")
    return documents


def main():
    """
    Main function to load, process, and store documents.
    """
    logging.info("Starting the document processing pipeline.")

    # Connect to MongoDB
    logging.info("Connecting to MongoDB.")
    client = connect_to_mongodb()
    collection = client[DATABASE_NAME][COLLECTION_NAME]

    # Load all documents from the dataset
    docs = load_all_documents("trec-covid")
    logging.info(f"Loaded {len(docs)} documents from the dataset.")

    # Choose the chunker implementation (TextSplitter or SemanticChunker)
    use_text_splitter = True  # Set to False to use SemanticChunker
    if use_text_splitter:
        chunker = TextSplitterWrapper(chunk_size=1000, chunk_overlap=200)
    else:
        chunker = SemanticChunkerWrapper(embedding_wrapper)

    # Process and store the documents
    processor = DocumentProcessor(chunker, embedding_wrapper, collection)
    processor.process_documents(docs, batch_size=50)

    logging.info("Document processing pipeline completed successfully.")


if __name__ == "__main__":
    main()
