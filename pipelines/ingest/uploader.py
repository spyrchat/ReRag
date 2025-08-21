"""
Vector store uploader with idempotent operations and versioning.
Handles dense, sparse, and hybrid vector upserts to Qdrant.
"""
import uuid
import logging
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, SparseVectorParams, Distance
from qdrant_client.http.models import SparseVector
from qdrant_client import models as qmodels

from database.qdrant_controller import QdrantVectorDB
from pipelines.contracts import ChunkMeta, IngestionRecord


logger = logging.getLogger(__name__)


def string_to_uuid(text: str) -> str:
    """Convert a string to a deterministic UUID."""
    # Create a deterministic UUID from string using SHA256
    hash_bytes = hashlib.sha256(text.encode('utf-8')).digest()[:16]
    return str(uuid.UUID(bytes=hash_bytes))


class VectorStoreUploader:
    """Handles idempotent vector store uploads with versioning."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.qdrant_config = config.get("qdrant", {})
        
        # Collection configuration
        self.collection_name = self.qdrant_config.get("collection", "default_collection")
        self.dense_vector_name = self.qdrant_config.get("dense_vector_name", "dense")
        self.sparse_vector_name = self.qdrant_config.get("sparse_vector_name", "sparse")
        
        # Upload configuration
        self.batch_size = config.get("upload", {}).get("batch_size", 100)
        self.wait_for_completion = config.get("upload", {}).get("wait", True)
        self.enable_versioning = config.get("upload", {}).get("versioning", True)
        
        # Initialize Qdrant
        self.vector_db = QdrantVectorDB()
        self.client = self.vector_db.get_client()
        
        logger.info(f"Initialized uploader for collection: {self.collection_name}")
    
    def upload_chunks(self, chunk_metas: List[ChunkMeta]) -> IngestionRecord:
        """Upload chunk metas to vector store with full lineage tracking."""
        if not chunk_metas:
            logger.warning("No chunks provided for upload")
            return self._create_empty_record()
        
        logger.info(f"Starting upload of {len(chunk_metas)} chunks to {self.collection_name}")
        
        # Create ingestion record
        record = IngestionRecord(
            dataset_name=chunk_metas[0].source,
            dataset_version=chunk_metas[0].dataset_version,
            config_hash=chunk_metas[0].config_hash or "unknown",
            git_commit=chunk_metas[0].git_commit,
            total_documents=len(set(meta.doc_id for meta in chunk_metas)),
            total_chunks=len(chunk_metas),
            successful_chunks=0,
            failed_chunks=0,
            started_at=datetime.utcnow()
        )
        
        # Prepare collection
        self._ensure_collection_exists(chunk_metas)
        
        # Upload in batches
        successful_count = 0
        failed_count = 0
        
        for i in range(0, len(chunk_metas), self.batch_size):
            batch = chunk_metas[i:i + self.batch_size]
            batch_success, batch_failed = self._upload_batch(batch)
            successful_count += batch_success
            failed_count += batch_failed
            
            logger.info(f"Batch {i//self.batch_size + 1}: {batch_success} successful, {batch_failed} failed")
        
        # Update record
        record.successful_chunks = successful_count
        record.failed_chunks = failed_count
        record.mark_complete()
        
        # Add sample IDs for verification
        record.sample_doc_ids = list(set(meta.doc_id for meta in chunk_metas[:10]))
        record.sample_chunk_ids = [meta.chunk_id for meta in chunk_metas[:10]]
        
        logger.info(f"Upload completed: {successful_count} successful, {failed_count} failed")
        return record
    
    def _upload_batch(self, chunk_metas: List[ChunkMeta]) -> tuple[int, int]:
        """Upload a batch of chunk metas."""
        points = []
        successful_count = 0
        failed_count = 0
        
        for meta in chunk_metas:
            try:
                point = self._chunk_meta_to_point(meta)
                if point:
                    points.append(point)
                    successful_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"Error converting chunk {meta.chunk_id} to point: {e}")
                failed_count += 1
        
        # Upsert points
        if points:
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=self.wait_for_completion
                )
                logger.debug(f"Successfully upserted {len(points)} points")
            except Exception as e:
                logger.error(f"Error upserting batch: {e}")
                # All points in batch failed
                failed_count += successful_count
                successful_count = 0
        
        return successful_count, failed_count
    
    def _chunk_meta_to_point(self, meta: ChunkMeta) -> Optional[PointStruct]:
        """Convert ChunkMeta to Qdrant PointStruct."""
        try:
            # Extract embeddings from ChunkMeta fields
            dense_embedding = meta.dense_embedding
            sparse_embedding = meta.sparse_embedding
            
            # Prepare vectors
            vectors = {}
            
            if dense_embedding:
                vectors[self.dense_vector_name] = dense_embedding
            
            if sparse_embedding:
                # Handle different sparse formats
                if isinstance(sparse_embedding, dict):
                    # Convert {token_id: weight} to SparseVector
                    indices = list(sparse_embedding.keys())
                    values = list(sparse_embedding.values())
                    vectors[self.sparse_vector_name] = SparseVector(
                        indices=indices,
                        values=values
                    )
                elif hasattr(sparse_embedding, 'indices') and hasattr(sparse_embedding, 'values'):
                    # Already in correct format
                    vectors[self.sparse_vector_name] = sparse_embedding
            
            # Prepare payload (exclude embeddings to save space)
            payload = meta.to_dict()
            payload.pop("metadata", None)  # Remove nested metadata
            
            # Add computed fields
            payload["embedding_types"] = list(vectors.keys())
            payload["has_dense"] = self.dense_vector_name in vectors
            payload["has_sparse"] = self.sparse_vector_name in vectors
            
            return PointStruct(
                id=string_to_uuid(meta.chunk_id),  # Convert string ID to UUID
                vector=vectors,
                payload=payload
            )
            
        except Exception as e:
            logger.error(f"Error creating point for chunk {meta.chunk_id}: {e}")
            return None
    
    def _ensure_collection_exists(self, chunk_metas: List[ChunkMeta]):
        """Ensure collection exists with proper configuration."""
        if self.client.collection_exists(self.collection_name):
            logger.info(f"Collection {self.collection_name} already exists")
            return
        
        # Determine vector dimensions
        dense_dim = None
        for meta in chunk_metas:
            dense_embedding = meta.dense_embedding
            if dense_embedding:
                dense_dim = len(dense_embedding)
                break
        
        if dense_dim is None:
            logger.warning("No dense embeddings found, using default dimension")
            dense_dim = 384  # Default dimension
        
        # Create collection with vector configurations
        vectors_config = {}
        sparse_vectors_config = {}
        
        # Add dense vector config if needed
        if any(meta.dense_embedding for meta in chunk_metas):
            vectors_config[self.dense_vector_name] = VectorParams(
                size=dense_dim,
                distance=Distance.COSINE
            )
        
        # Add sparse vector config if needed
        if any(meta.sparse_embedding for meta in chunk_metas):
            sparse_vectors_config[self.sparse_vector_name] = SparseVectorParams(
                index=qmodels.SparseIndexParams(on_disk=False)
            )
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config
        )
        
        logger.info(f"Created collection {self.collection_name} with {len(vectors_config)} dense and {len(sparse_vectors_config)} sparse vectors")
    
    def _create_empty_record(self) -> IngestionRecord:
        """Create empty ingestion record for failed cases."""
        return IngestionRecord(
            dataset_name="unknown",
            dataset_version="unknown",
            config_hash="unknown",
            git_commit=None,
            total_documents=0,
            total_chunks=0,
            successful_chunks=0,
            failed_chunks=0,
            started_at=datetime.utcnow()
        )
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "points_count": info.points_count,
                "vectors_config": info.config.params.vectors,
                "sparse_vectors_config": info.config.params.sparse_vectors,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    def verify_upload(self, sample_chunk_ids: List[str]) -> Dict[str, Any]:
        """Verify that sample chunks were uploaded correctly."""
        verification_results = {
            "total_samples": len(sample_chunk_ids),
            "found_samples": 0,
            "missing_samples": [],
            "verification_passed": False
        }
        
        try:
            for chunk_id in sample_chunk_ids:
                # Convert chunk_id to UUID (same as during upload)
                uuid_id = string_to_uuid(chunk_id)
                points = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[uuid_id]
                )
                
                if points:
                    verification_results["found_samples"] += 1
                else:
                    verification_results["missing_samples"].append(chunk_id)
            
            # Consider verification passed if at least 90% of samples found
            success_rate = verification_results["found_samples"] / verification_results["total_samples"]
            verification_results["verification_passed"] = success_rate >= 0.9
            verification_results["success_rate"] = success_rate
            
        except Exception as e:
            logger.error(f"Error during verification: {e}")
            verification_results["error"] = str(e)
        
        return verification_results
