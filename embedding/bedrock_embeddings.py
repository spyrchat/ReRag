import json
import uuid
import boto3
import logging
import requests
from typing import List, Optional
from botocore.exceptions import ClientError
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from langchain_core.embeddings import Embeddings
from embedding.utils import write_jsonl, upload_to_s3

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TitanEmbedder(Embeddings):
    """
    LangChain-compatible embedder for Amazon Titan (via Bedrock).
    Supports both real-time and batch embedding operations.
    """

    def __init__(self, model: str = "amazon.titan-embed-text-v2:0", region: str = "us-east-1"):
        """
        Initialize the TitanEmbedder.

        Args:
            model (str): Titan model ID.
            region (str): AWS region for Bedrock and S3.
        """
        self.model = model
        self.region = region
        self.realtime_client = boto3.client(
            "bedrock-runtime", region_name=region)
        self.batch_client = boto3.client("bedrock", region_name=region)
        self.s3 = boto3.client("s3", region_name=region)
        logger.info(
            f"Initialized TitanEmbedder with model: {model}, region: {region}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Real-time embedding of multiple documents.

        Args:
            texts (List[str]): List of strings to embed.

        Returns:
            List[List[float]]: Corresponding list of embedding vectors.
        """
        logger.info(f"Starting real-time embedding for {len(texts)} texts.")
        embeddings = []
        for i, text in enumerate(texts):
            try:
                embedding = self._embed_single(text)
                embeddings.append(embedding)
                if i % 100 == 0 and i > 0:
                    logger.info(f"Embedded {i} texts...")
            except Exception as e:
                logger.error(f"Embedding failed at index {i}: {e}")
                embeddings.append([0.0] * 1024)  # fallback vector
        logger.info("Real-time embedding complete.")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Real-time embedding of a single query string.

        Args:
            text (str): Query text to embed.

        Returns:
            List[float]: Embedding vector.
        """
        return self._embed_single(text)

    def _embed_single(self, text: str) -> List[float]:
        """
        Internal helper for real-time embedding of a single string.

        Args:
            text (str): Input text.

        Returns:
            List[float]: Embedding vector.
        """
        body = {
            "inputText": text,
            "normalize": True  # Use model default dimension
        }
        try:
            response = self.realtime_client.invoke_model(
                body=json.dumps(body),
                modelId=self.model,
                accept="application/json",
                contentType="application/json"
            )
            parsed = json.loads(response["body"].read())
            return parsed["embedding"]
        except ClientError as e:
            logger.error(f"AWS Bedrock client error: {e}")
            raise
        except Exception as e:
            logger.error(f"General error during Titan embed: {e}")
            raise

    def start_batch_job_from_texts(
        self,
        texts: List[str],
        s3_bucket: str,
        s3_output_uri: str,
        role_arn: str
    ) -> str:
        """
        Create and start a batch job from a list of texts.

        Args:
            texts (List[str]): Input texts to embed.
            s3_bucket (str): S3 bucket to upload the input file to.
            s3_output_uri (str): S3 URI to store the output embeddings.
            role_arn (str): IAM role ARN for Bedrock batch job.

        Returns:
            str: ARN of the started job.
        """
        job_id = uuid.uuid4().hex
        local_path = f"sandbox/titan_batch_input_{job_id}.jsonl"
        s3_key = f"batch-input/{job_id}.jsonl"
        s3_input_uri = f"s3://{s3_bucket}/{s3_key}"

        logger.info(f"Writing input JSONL to: {local_path}")
        write_jsonl(texts, local_path)

        logger.info(f"Uploading input file to S3: {s3_input_uri}")
        upload_to_s3(local_path, s3_bucket, s3_key)

        return self.start_batch_job(
            s3_input_uri=s3_input_uri,
            s3_output_uri=s3_output_uri,
            role_arn=role_arn,
            job_name=f"titan-batch-{job_id}"
        )

    def start_batch_job(
        self,
        s3_input_uri: str,
        s3_output_uri: str,
        role_arn: str,
        job_name: Optional[str] = None
    ) -> str:
        """
        Submit a Titan batch embedding job using SigV4-signed HTTP request.

        Args:
            s3_input_uri (str): S3 URI for input JSONL.
            s3_output_uri (str): S3 URI for output embeddings.
            role_arn (str): IAM role ARN with Bedrock permissions.
            job_name (Optional[str]): Optional job name.

        Returns:
            str: Job ARN if submitted successfully.
        """
        logger.info(f"Submitting Titan batch job: {job_name}")
        logger.info(f"Input URI: {s3_input_uri}")
        logger.info(f"Output URI: {s3_output_uri}")
        logger.info(f"Role ARN: {role_arn}")

        payload = {
            "jobName": job_name,
            "clientRequestToken": job_name,
            "modelId": self.model,
            "inputDataConfig": {
                "s3InputDataConfig": {
                    "s3Uri": s3_input_uri
                }
            },
            "outputDataConfig": {
                "s3OutputDataConfig": {
                    "s3Uri": s3_output_uri
                }
            },
            "roleArn": role_arn
        }

        # Manually signed HTTP request with SigV4
        credentials = boto3.Session().get_credentials().get_frozen_credentials()
        endpoint = f"https://bedrock.{self.region}.amazonaws.com/model-invocation-job"
        headers = {
            "Host": f"bedrock.{self.region}.amazonaws.com",
            "Content-Type": "application/json"
        }

        request = AWSRequest(
            method="POST",
            url=endpoint,
            data=json.dumps(payload),
            headers=headers
        )
        SigV4Auth(credentials, "bedrock", self.region).add_auth(request)

        response = requests.post(
            url=endpoint,
            data=request.body,
            headers=dict(request.headers)
        )

        if response.status_code == 200:
            job_arn = response.json()["jobArn"]
            logger.info(f"Titan batch job started: {job_arn}")
            return job_arn
        else:
            logger.error(
                f"Failed to start job: {response.status_code} {response.text}")
            raise RuntimeError(
                f"Failed to start Titan batch job: {response.status_code} - {response.text}"
            )
