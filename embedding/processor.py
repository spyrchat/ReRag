from langchain.schema import Document
from embedding.base_embedder import BaseEmbedder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Tuple, List, Optional
import uuid
import os
from dotenv import load_dotenv
from .utils import write_jsonl, upload_to_s3, download_from_s3, parse_jsonl_embeddings
import boto3
import time
import logging

load_dotenv()
logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    def __init__(
        self,
        dense_embedder: BaseEmbedder,
        splitter: RecursiveCharacterTextSplitter,
        sparse_embedder: Optional[BaseEmbedder] = None,
    ):
        """
        Initialize the pipeline with dense (required) and optional sparse embedders.
        """
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder
        self.splitter = splitter

    def run(self, docs: List[Document], use_batch: bool = False) -> Tuple[List[str], Optional[List[List[float]]], Optional[List[List[float]]]]:
        """
        Embeds documents using dense (and optionally sparse) strategies.

        Returns:
            texts: List of chunked texts.
            dense_vectors: Dense embeddings or None.
            sparse_vectors: Sparse embeddings or None.
        """
        chunks = self.splitter.split_documents(docs)
        texts = [doc.page_content for doc in chunks]

        dense_vectors = None
        sparse_vectors = None

        # Handle dense embedding
        if use_batch:
            dense_vectors = self._run_batch_dense(texts)
        else:
            dense_vectors = self.dense_embedder.embed_documents(texts)

        # Handle sparse embedding
        if self.sparse_embedder:
            logger.info("Starting sparse embedding...")
            sparse_vectors = self.sparse_embedder.embed_documents(texts)

        return texts, dense_vectors, sparse_vectors

    def _run_batch_dense(self, texts: List[str]) -> List[List[float]]:
        """
        Internal method to run batch dense embedding.
        """
        s3_bucket = os.getenv("TITAN_S3_BUCKET")
        s3_output_uri = os.getenv("TITAN_S3_OUTPUT_URI")
        role_arn = os.getenv("TITAN_ROLE_ARN")

        if not all([s3_bucket, s3_output_uri, role_arn]):
            raise ValueError(
                "Missing required batch job configuration in .env")

        local_path = f"sandbox/titan_batch_input_{uuid.uuid4().hex}.jsonl"
        s3_key = f"batch-input/{uuid.uuid4().hex}.jsonl"
        s3_input_uri = f"s3://{s3_bucket}/{s3_key}"

        write_jsonl(texts, local_path)
        upload_to_s3(local_path, s3_bucket, s3_key)

        if hasattr(self.dense_embedder, "start_batch_job"):
            job_arn = self.dense_embedder.start_batch_job(
                s3_input_uri=s3_input_uri,
                s3_output_uri=s3_output_uri,
                role_arn=role_arn
            )
            logger.info(f"Titan batch job started: {job_arn}")

            job_id = job_arn.split("/")[-1]
            bedrock = boto3.client(
                "bedrock", region_name=self.dense_embedder.region)
            logger.info("Polling for batch job completion...")
            while True:
                response = bedrock.get_model_invocation_job(
                    jobIdentifier=job_id)
                status = response["status"]
                if status in ("COMPLETED", "FAILED"):
                    break
                logger.info(f"Current job status: {status}. Waiting...")
                time.sleep(10)

            if status == "FAILED":
                raise RuntimeError("Batch job failed.")

            output_s3_uri = response["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
            logger.info(
                f"Job completed. Fetching results from: {output_s3_uri}")

            output_file = download_from_s3(output_s3_uri)
            vectors = parse_jsonl_embeddings(output_file)
            return vectors
        else:
            raise NotImplementedError(
                "Embedder does not support batch job execution."
            )
