import os
import logging
from typing import List, Dict
from pymongo import MongoClient
from pymongo.collection import Collection
from dotenv import load_dotenv
from embeddings import get_titan_embedding


def load_env() -> None:
    """Load environment variables from a .env file."""
    load_dotenv()


def get_mongodb_uri() -> str:
    """Retrieve the MongoDB URI from environment variables."""
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise EnvironmentError(
            "MONGODB_URI is not defined in the environment.")
    return uri


def connect_to_mongodb() -> MongoClient:
    """
    Connect to MongoDB using the URI defined in environment variables.

    Returns:
        MongoClient: A connected MongoDB client instance.
    """
    load_env()
    uri = get_mongodb_uri()
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")  # Test the connection
        logging.info("Successfully connected to MongoDB.")
        return client
    except Exception as e:
        logging.error("Failed to connect to MongoDB.")
        raise ConnectionError(f"Failed to connect to MongoDB: {e}")


class MongoAtlasRetriever:
    def __init__(self, collection: Collection, embedding_wrapper, index_name: str = "vector-index", top_k: int = 5):
        """
        Args:
            collection (Collection): pymongo collection instance
            embedding_wrapper: An embedding wrapper instance (e.g., TitanEmbeddingWrapper).
            index_name (str): Atlas Search vector index name.
            top_k (int): Number of top results to return.
        """
        self.collection = collection
        self.embedding_wrapper = embedding_wrapper
        self.index_name = index_name
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve top-k documents for a given query.

        Args:
            query (str): The text query.

        Returns:
            List[Dict]: List of documents with similarity scores.
        """
        # Generate the query vector using the provided embedding wrapper
        logging.info("Generating query vector using the embedding wrapper.")
        query_vector = self.embedding_wrapper.embed_documents([query])[0]

        # MongoDB Atlas Search pipeline for vector search
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.index_name,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": 100,
                    "limit": self.top_k
                }
            }
        ]

        logging.info(f"Running vector search with top_k={self.top_k}.")
        return list(self.collection.aggregate(pipeline))


def normalize_vector_score(score: float, min_val: float = 0.965, max_val: float = 1.0) -> float:
    """
    Normalize a similarity score from vector search to a 0-1 scale.

    Args:
        score (float): Original similarity score.
        min_val (float): Minimum expected value.
        max_val (float): Maximum expected value.

    Returns:
        float: Normalized score.
    """
    return max(0.0, min(1.0, (score - min_val) / (max_val - min_val)))


def insert_documents(collection: Collection, documents: List[dict]) -> None:
    """
    Insert a list of documents into a MongoDB collection.

    Args:
        collection (Collection): A MongoDB collection object.
        documents (List[dict]): List of documents to insert.

    Raises:
        ValueError: If documents list is empty.
    """
    if not documents:
        raise ValueError("No documents to insert.")
    collection.insert_many(documents)


def main():
    try:
        # Connect to MongoDB
        client = connect_to_mongodb()

        # Retrieve database and collection names from environment variables
        db_name = os.getenv("MONGODB_DATABASE", "aws_gen_ai")
        collection_name = os.getenv("MONGODB_COLLECTION", "NQ")

        # Access the database and collection
        db = client[db_name]
        collection = db[collection_name]

        # Sample documents to insert
        sample_documents = [
            {
                "doc_id": "doc1",
                "title": "Example Title 1",
                "text": "Example content for document 1.",
                "embedding": [0.1, 0.2, 0.3],
                "source": "test"
            },
            {
                "doc_id": "doc2",
                "title": "Example Title 2",
                "text": "Example content for document 2.",
                "embedding": [0.4, 0.5, 0.6],
                "source": "test"
            }
        ]

        # Insert documents into the collection
        insert_documents(collection, sample_documents)
        print(
            f"Successfully inserted {len(sample_documents)} documents into {collection.full_name}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
