from langchain.schema import Document
from embedding.base_embedder import BaseEmbedder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Tuple, List, Optional
import os
from dotenv import load_dotenv

load_dotenv()


class EmbeddingPipeline:
    def __init__(self, embedder: BaseEmbedder, splitter: RecursiveCharacterTextSplitter):
        self.embedder = embedder
        self.splitter = splitter

    def run(self, docs: List[Document], use_batch: bool = True, **batch_kwargs) -> Tuple[List[str], Optional[List[List[float]]]]:
        """
        If use_batch=True, initiate a batch job instead of returning embeddings.
        Otherwise, returns (texts, embeddings) as usual.
        """
        chunks = self.splitter.split(docs)
        texts = [doc.page_content for doc in chunks]

        if use_batch:
            if hasattr(self.embedder, "start_batch_job_from_texts"):
                s3_bucket = os.getenv("TITAN_S3_BUCKET")
                s3_output_uri = os.getenv("TITAN_S3_OUTPUT_URI")
                role_arn = os.getenv("TITAN_ROLE_ARN")

                if not all([s3_bucket, s3_output_uri, role_arn]):
                    raise ValueError("Missing S3 or IAM config in .env")

                job_arn = self.embedder.start_batch_job_from_texts(
                    texts=texts,
                    s3_bucket=s3_bucket,
                    s3_output_uri=s3_output_uri,
                    role_arn=role_arn
                )
                print(f"Started Titan batch job: {job_arn}")
                return texts, None
            else:
                raise NotImplementedError(
                    "Embedder does not support batch job execution")

        embeddings = self.embedder.embed_texts(texts)
        return texts, embeddings
