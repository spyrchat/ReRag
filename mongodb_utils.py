import os
from pymongo import MongoClient
from dotenv import load_dotenv


def load_env():
    """Loads environment variables from .env file."""
    load_dotenv()


def get_mongodb_uri() -> str:
    """Fetches MongoDB URI from environment."""
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI is missing in the .env file")
    return uri


def connect_to_mongodb():
    """
    Connects to MongoDB using URI from environment variables.

    Returns:
        MongoClient: Connected MongoDB client instance.
    """
    load_env()
    uri = get_mongodb_uri()

    try:
        client = MongoClient(uri)
        client.admin.command("ping")  # Check connectivity
        print("Connected to MongoDB successfully.")
        return client
    except Exception as e:
        print("Could not connect to MongoDB:", e)
        return None


def normalize_vector_search_score(score: float) -> float:
    """
    Normalizes a similarity score from vector search.

    Args:
        score (float): The original similarity score.

    Returns:
        float: The normalized score between 0 and 1.
    """
    min_val = 0.965
    max_val = 1.0
    return max(0.0, (score - min_val) / (max_val - min_val))


if __name__ == "__main__":
    # Optional: check individual parts of the connection string (debug only, don't print values)
    if os.getenv("MONGODB_CLUSTER"):
        print("Cluster name is set.")
    if os.getenv("MONGODB_USERNAME"):
        print("Username is set.")
    if os.getenv("MONGODB_PASSWORD"):
        print("Password is set.")

    client = connect_to_mongodb()

    if client:
        # Optionally select a database if known
        db_name = os.getenv("MONGODB_DB", "admin")
        db = client[db_name]
        print(f"Using database: {db.name}")
    else:
        print("Failed to connect to MongoDB.")
