import os
import glob
import json
import tqdm
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from constants import constants as c
from pymongo.database import Database
from pymongo.collection import Collection
from typing import Tuple, List, Any, Mapping, Dict
from pymongo. errors import ConnectionFailure
from pymongo.operations import SearchIndexModel

from events_dataset.events_dataset_utils import create_batches_of_filepaths


def connect_to_mongodb() -> (Tuple[None, None] |
                             Tuple[MongoClient[Mapping[str, Any] | Any], Database[Mapping[str, Any] | Any]]):
    """
    Set up a connection to the MongoDB database and returns the client and the database object. The credentials are loaded from
    the .env file.

    :return: A tuple with the MongoDB client and the database object
    """

    # Load the environment variables from the .env file
    load_dotenv()

    # Get the MongoDB credentials
    mongodb_username = os.getenv("MONGODB_USERNAME")
    mongodb_password = os.getenv("MONGODB_PASSWORD")
    mongodb_cluster  = os.getenv("MONGODB_CLUSTER")

    # Construct the MongoDB URI
    mongodb_uri = f"mongodb+srv://{mongodb_username}:{mongodb_password}@{mongodb_cluster}.mongodb.net/"

    # Connect to MongoDB
    mongo_client = MongoClient(mongodb_uri)

    # Check if the connection is successful
    mongo_client.admin.command('ping')

    return mongo_client, mongo_client[c.AWS_GEN_AI]


def normalize_vector_search_score(score: float) -> float:
    """
    A method to normalize the vector search score.

    :param score: The score to normalize

    :return: The normalized score
    """

    min_val = 0.965
    max_val = 1.0

    return max(0.0, ((score - min_val) / (max_val - min_val)))
