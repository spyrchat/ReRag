import boto3
import json
from typing import List


class TitanEmbeddingWrapper:
    def __init__(self, model: str, region: str = "us-east-1"):
        """
        Wrapper for generating embeddings using Amazon Titan Embed Text v2.

        Args:
            model (str): The model ID to use for embedding.
            region (str): AWS region (default: us-east-1).
        """
        self.model = model
        self.region = region

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using the Titan embedding model.

        Args:
            texts (List[str]): List of input texts.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        embeddings = []
        for text in texts:
            try:
                embedding = get_titan_embedding(
                    text, model=self.model, region=self.region)
                embeddings.append(embedding)
            except Exception as e:
                print(
                    f"Failed to generate embedding for text: {text[:30]}... Error: {e}")
                # Fallback to a zero vector if embedding fails
                embeddings.append([0.0] * 1024)
        return embeddings


def get_titan_embedding(text: str, model: str, region: str = "us-east-1") -> List[float]:
    """
    Generate a text embedding using Amazon Titan Embed Text v2.

    Args:
        text (str): The input text to embed.
        model (str): The model ID to use for embedding.
        region (str): AWS region (default: us-east-1).

    Returns:
        List[float]: The embedding vector as a list of floats.
    """
    try:
        # Initialize the Bedrock runtime client
        client = boto3.client("bedrock-runtime", region_name=region)

        # Prepare the request body
        body = {
            "inputText": text
        }

        # Invoke the model
        response = client.invoke_model(
            body=json.dumps(body),
            modelId=model,
            accept="application/json",
            contentType="application/json"
        )

        # Parse the response
        response_body = json.loads(response["body"].read())
        return response_body["embedding"]

    except Exception as e:
        raise Exception(f"Failed to generate embedding: {e}")


# Example usage
if __name__ == "__main__":
    text = "COVID-19 is caused by the SARS-CoV-2 virus."
    model_id = "amazon.titan-embed-text-v2:0"

    # Using the wrapper
    wrapper = TitanEmbeddingWrapper(model=model_id)
    embeddings = wrapper.embed_documents([text])

    print("Embedding vector (first 10 values):", embeddings[0][:10], "...")
    print("Embedding dimension:", len(embeddings[0]))  # Should be 1536
