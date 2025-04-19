import json
import uuid
from typing import List, Dict, Any
import boto3


def batchify(items, batch_size):
    """
    Splits a list of items into smaller batches of a specified size.

    Args:
        items (list): The list of items to be split into batches.
        batch_size (int): The maximum number of items in each batch.
    """
    for i in range(0, len(items), batch_size):
        # Yield a slice of the list from index `i` to `i + batch_size`
        yield items[i:i + batch_size]


def write_jsonl(texts: List[str], filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps({"inputText": text}) + "\n")


def upload_to_s3(local_path: str, s3_bucket: str, s3_key: str):
    s3 = boto3.client("s3")
    s3.upload_file(local_path, s3_bucket, s3_key)
    return f"s3://{s3_bucket}/{s3_key}"
