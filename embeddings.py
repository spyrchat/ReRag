import boto3
import json


def get_titan_embedding(text: str, model: str, region: str = "us-east-1") -> list:
    """
    Generate a text embedding using Amazon Titan Embed Text v2.

    Args:
        text (str): The input text to embed.
        region (str): AWS region (default: us-east-1).

    Returns:
        list: The embedding vector as a list of floats.
    """
    client = boto3.client("bedrock-runtime", region_name=region)

    body = {
        "inputText": text
    }

    response = client.invoke_model(
        body=json.dumps(body),
        modelId=model,
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response["body"].read())
    return response_body["embedding"]


# Example usage
if __name__ == "__main__":
    text = "COVID-19 is caused by the SARS-CoV-2 virus."
    embedding = get_titan_embedding(text, model="amazon.titan-embed-text-v2:0")
    print("Embedding vector:", embedding[:10], "...")  # print first 10 values
    print("Embedding dimension:", len(embedding))       # should be 1536
