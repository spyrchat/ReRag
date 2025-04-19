import json
from typing import List
import boto3
from botocore.exceptions import ClientError

from embedding.base_embedder import BaseEmbedder
from embedding.utils import batchify


class TitanEmbedder(BaseEmbedder):
    def __init__(self, model: str = "amazon.titan-embed-text-v2:0", region: str = "us-east-1", batch_size: int = 16):
        self.model = model
        self.region = region
        self.batch_size = batch_size
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for batch in batchify(texts, self.batch_size):
            try:
                batch_embeddings = self._embed_batch(batch)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                raise RuntimeError(f"Batch embedding failed: {e}")
        return embeddings

    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        body = {
            # Titan accepts str or List[str]
            "inputText": batch if len(batch) > 1 else batch[0]
        }

        try:
            response = self.client.invoke_model(
                body=json.dumps(body),
                modelId=self.model,
                accept="application/json",
                contentType="application/json"
            )
            parsed = json.loads(response["body"].read())

            # Titan returns "embedding" for single input or "embeddings" for batch
            if isinstance(parsed, dict) and "embeddings" in parsed:
                return parsed["embeddings"]
            elif isinstance(parsed, dict) and "embedding" in parsed:
                return [parsed["embedding"]]
            else:
                raise ValueError(f"Unexpected Titan response format: {parsed}")

        except ClientError as e:
            raise RuntimeError(f"AWS error: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to call Titan: {e}")
