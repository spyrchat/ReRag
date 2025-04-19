import json
import uuid
import os
from typing import List, Dict, Any
import boto3


def batchify(items: List[Any], batch_size: int):
    """
    Splits a list of items into smaller batches of a specified size.

    Args:
        items (list): The list of items to be split into batches.
        batch_size (int): The maximum number of items in each batch.

    Yields:
        List[Any]: A batch of items.
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def write_jsonl(texts: List[str], filepath: str):
    """
    Writes a list of strings to a JSONL file, where each line is a JSON object with a single key: 'inputText'.

    Args:
        texts (List[str]): The list of input texts to write.
        filepath (str): The local file path where to save the JSONL.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps({"inputText": text}) + "\n")


def upload_to_s3(local_path: str, s3_bucket: str, s3_key: str) -> str:
    """
    Uploads a local file to the specified S3 bucket and key.

    Args:
        local_path (str): Local file path.
        s3_bucket (str): S3 bucket name.
        s3_key (str): Key (path) in the bucket.

    Returns:
        str: The full S3 URI of the uploaded file.
    """
    s3 = boto3.client("s3")
    s3.upload_file(local_path, s3_bucket, s3_key)
    return f"s3://{s3_bucket}/{s3_key}"


def download_from_s3(s3_uri: str, local_dir: str = "sandbox") -> str:
    """
    Downloads a file from S3 URI to a local path.

    Args:
        s3_uri (str): The full S3 URI (e.g., s3://bucket/path/to/file.jsonl)
        local_dir (str): Directory to download the file into (default: sandbox)

    Returns:
        str: The local file path of the downloaded file.
    """
    s3 = boto3.client("s3")
    if not s3_uri.startswith("s3://"):
        raise ValueError("Invalid S3 URI")

    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    filename = key.split("/")[-1]
    local_path = os.path.join(local_dir, filename)

    os.makedirs(local_dir, exist_ok=True)
    s3.download_file(bucket, key, local_path)
    return local_path


def parse_jsonl_embeddings(filepath: str) -> List[List[float]]:
    """
    Parses a JSONL file with lines containing { \"embedding\": [...] } and returns a list of embeddings.

    Args:
        filepath (str): Path to the JSONL file.

    Returns:
        List[List[float]]: Parsed list of embedding vectors.
    """
    embeddings = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "embedding" in obj:
                embeddings.append(obj["embedding"])
    return embeddings
