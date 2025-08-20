"""
Main ingestion pipeline that orchestrates all components.
Implements the complete theory-backed pipeline with lineage tracking.
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from pipelines.contracts import (
    DatasetAdapter, ChunkMeta, IngestionRecord, DatasetSplit
)
from pipelines.ingest.validator import DocumentValidator
from pipelines.ingest.chunker import ChunkingStrategyFactory
from pipelines.ingest.embedder import EmbeddingPipeline
from pipelines.ingest.uploader import VectorStoreUploader
from pipelines.ingest.smoke_tests import SmokeTestRunner
from config.config_loader import load_config


logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Main ingestion pipeline orchestrating all components.
    Implements deterministic IDs, idempotent loads, and comprehensive lineage.
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize pipeline with configuration."""
        if config:
            self.config = config
        elif config_path:
            self.config = load_config(config_path)
        else:
            self.config = load_config()  # Default config.yml
        
        # Initialize components
        self.validator = DocumentValidator(self.config.get("validation", {}))
        self.embedding_pipeline = EmbeddingPipeline(self.config)
        self.uploader = VectorStoreUploader(self.config)
        self.smoke_test_runner = SmokeTestRunner(self.config)
        
        # Pipeline configuration
        self.dry_run = False
        self.max_documents = None
        self.canary_mode = False
        
        # Output directories
        self.output_dir = Path(self.config.get("output_dir", "output"))
        self.lineage_dir = self.output_dir / "lineage"
        self.lineage_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Ingestion pipeline initialized")
    
    def ingest_dataset(
        self, 
        adapter: DatasetAdapter, 
        split: DatasetSplit = DatasetSplit.ALL,
        dry_run: bool = False,
        max_documents: Optional[int] = None,
        canary_mode: bool = False
    ) -> IngestionRecord:
        """
        Main ingestion method - processes a complete dataset.
        
        Args:
            adapter: Dataset adapter implementing DatasetAdapter interface
            split: Dataset split to process
            dry_run: If True, don't upload to vector store
            max_documents: Limit number of documents (for testing)
            canary_mode: If True, use canary collection name
        """
        self.dry_run = dry_run
        self.max_documents = max_documents
        self.canary_mode = canary_mode
        
        logger.info(f"Starting ingestion: {adapter.source_name} v{adapter.version} (split: {split.value})")
        if dry_run:
            logger.info("DRY RUN MODE - No uploads will be performed")
        if canary_mode:
            logger.info("CANARY MODE - Using canary collection")
        
        # Create ingestion record
        record = IngestionRecord(
            dataset_name=adapter.source_name,
            dataset_version=adapter.version,
            config_hash=self._compute_config_hash(),
            git_commit=self._get_git_commit(),
            total_documents=0,
            total_chunks=0,
            successful_chunks=0,
            failed_chunks=0,
            started_at=datetime.utcnow(),
            chunk_strategy=self.config.get("chunking", {}),
            embedding_strategy=self.config.get("embedding", {})
        )
        
        try:
            # Step 1: Read and validate documents
            logger.info("Step 1: Reading and validating documents...")
            documents = self._read_and_validate_documents(adapter, split, record)
            
            if not documents:
                logger.warning("No valid documents found")
                record.mark_complete()
                return record
            
            # Step 2: Chunk documents
            logger.info("Step 2: Chunking documents...")
            chunks = self._chunk_documents(documents, record)
            
            # Step 3: Generate embeddings and create ChunkMeta
            logger.info("Step 3: Generating embeddings...")
            chunk_metas = self._process_chunks(chunks, record)
            
            # Step 4: Upload to vector store (unless dry run)
            if not dry_run:
                logger.info("Step 4: Uploading to vector store...")
                upload_record = self._upload_chunks(chunk_metas)
                
                # Update record with upload results
                record.successful_chunks = upload_record.successful_chunks
                record.failed_chunks = upload_record.failed_chunks
                
                # Step 5: Run smoke tests
                logger.info("Step 5: Running smoke tests...")
                smoke_results = self._run_smoke_tests()
                record.metadata = {"smoke_test_results": smoke_results}
                
            else:
                logger.info("Step 4: Skipped (dry run)")
                record.successful_chunks = len(chunk_metas)
                record.failed_chunks = 0
            
            # Complete and save record
            record.mark_complete()
            self._save_lineage(record)
            
            logger.info(f"Ingestion completed: {record.successful_chunks} successful, {record.failed_chunks} failed")
            return record
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            record.mark_complete()
            record.metadata = {"error": str(e)}
            self._save_lineage(record)
            raise
    
    def _read_and_validate_documents(
        self, 
        adapter: DatasetAdapter, 
        split: DatasetSplit,
        record: IngestionRecord
    ) -> List[Any]:
        """Read documents from adapter and validate them."""
        # Read rows from adapter
        rows = list(adapter.read_rows(split))
        
        if self.max_documents:
            rows = rows[:self.max_documents]
            logger.info(f"Limited to {len(rows)} documents for testing")
        
        # Convert to documents
        documents = adapter.to_documents(rows, split)
        record.total_documents = len(documents)
        
        logger.info(f"Read {len(documents)} documents from {adapter.source_name}")
        
        # Validate documents
        validation_results = self.validator.validate_batch(documents)
        
        # Filter valid documents and clean them
        valid_documents = []
        validation_errors = []
        
        for doc, validation_result in zip(documents, validation_results):
            if validation_result.valid:
                cleaned_doc = self.validator.clean_document(doc)
                valid_documents.append(cleaned_doc)
            else:
                validation_errors.append({
                    "doc_id": validation_result.doc_id,
                    "errors": validation_result.errors
                })
        
        logger.info(f"Validation: {len(valid_documents)} valid, {len(validation_errors)} invalid")
        
        if validation_errors:
            logger.warning(f"Validation errors found: {validation_errors[:5]}")  # Show first 5
        
        return valid_documents
    
    def _chunk_documents(self, documents: List[Any], record: IngestionRecord) -> List[Any]:
        """Chunk documents using configured strategy."""
        chunking_config = self.config.get("chunking", {})
        strategy_name = chunking_config.get("strategy", "recursive")
        
        # Auto-select strategy if needed
        if strategy_name == "auto":
            # Analyze first document to determine strategy
            sample_content = documents[0].page_content if documents else ""
            strategy = ChunkingStrategyFactory.get_strategy_for_content(sample_content, chunking_config)
        else:
            strategy = ChunkingStrategyFactory.create_strategy(strategy_name, chunking_config)
        
        logger.info(f"Using chunking strategy: {strategy.strategy_name}")
        
        # Chunk all documents
        chunks = strategy.chunk(documents)
        
        # Update chunk metadata with document totals
        doc_chunk_counts = {}
        for chunk in chunks:
            doc_id = chunk.metadata.get("external_id", "unknown")
            doc_chunk_counts[doc_id] = doc_chunk_counts.get(doc_id, 0) + 1
        
        # Update num_chunks for each chunk
        for chunk in chunks:
            doc_id = chunk.metadata.get("external_id", "unknown")
            chunk.metadata["num_chunks"] = doc_chunk_counts.get(doc_id, 1)
        
        record.total_chunks = len(chunks)
        logger.info(f"Generated {len(chunks)} chunks from {len(documents)} documents")
        
        return chunks
    
    def _process_chunks(self, chunks: List[Any], record: IngestionRecord) -> List[ChunkMeta]:
        """Process chunks through embedding pipeline."""
        # Generate embeddings and convert to ChunkMeta
        chunk_metas = self.embedding_pipeline.process_documents(chunks)
        
        # Add sample IDs to record
        record.sample_doc_ids = list(set(meta.doc_id for meta in chunk_metas[:10]))
        record.sample_chunk_ids = [meta.chunk_id for meta in chunk_metas[:10]]
        
        logger.info(f"Processed {len(chunk_metas)} chunk metas with embeddings")
        return chunk_metas
    
    def _upload_chunks(self, chunk_metas: List[ChunkMeta]) -> IngestionRecord:
        """Upload chunks to vector store."""
        if self.canary_mode:
            # Modify collection name for canary
            original_collection = self.uploader.collection_name
            self.uploader.collection_name = f"{original_collection}_canary"
            logger.info(f"Using canary collection: {self.uploader.collection_name}")
        
        upload_record = self.uploader.upload_chunks(chunk_metas)
        
        # Verify upload with sample
        if chunk_metas:
            sample_ids = [meta.chunk_id for meta in chunk_metas[:5]]
            verification = self.uploader.verify_upload(sample_ids)
            upload_record.metadata = {"verification": verification}
            
            if not verification.get("verification_passed", False):
                logger.warning(f"Upload verification failed: {verification}")
        
        return upload_record
    
    def _run_smoke_tests(self) -> Dict[str, Any]:
        """Run post-ingestion smoke tests."""
        return self.smoke_test_runner.run_all_tests()
    
    def _save_lineage(self, record: IngestionRecord):
        """Save ingestion lineage for reproducibility."""
        lineage_file = self.lineage_dir / f"{record.dataset_name}_{record.run_id}.json"
        
        lineage_data = {
            "ingestion_record": record.dict(),
            "config": self.config,
            "environment": {
                "python_version": self._get_python_version(),
                "git_commit": record.git_commit,
                "timestamp": str(datetime.utcnow()),
                "hostname": os.uname().nodename if hasattr(os, 'uname') else "unknown"
            }
        }
        
        with open(lineage_file, 'w') as f:
            json.dump(lineage_data, f, indent=2, default=str)
        
        logger.info(f"Lineage saved to {lineage_file}")
    
    def _compute_config_hash(self) -> str:
        """Compute hash of current configuration."""
        import hashlib
        config_str = json.dumps(self.config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                    capture_output=True, text=True, cwd='.')
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def _get_python_version(self) -> str:
        """Get Python version."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def get_collection_status(self) -> Dict[str, Any]:
        """Get current collection status."""
        return self.uploader.get_collection_info()
    
    def cleanup_canary_collections(self):
        """Clean up canary collections after testing."""
        client = self.uploader.client
        collections = client.get_collections().collections
        
        canary_collections = [c.name for c in collections if "_canary" in c.name]
        
        for collection_name in canary_collections:
            try:
                client.delete_collection(collection_name)
                logger.info(f"Deleted canary collection: {collection_name}")
            except Exception as e:
                logger.error(f"Error deleting canary collection {collection_name}: {e}")


class BatchIngestionPipeline:
    """Pipeline for processing multiple datasets in sequence."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.pipeline = IngestionPipeline(config_path)
        self.results = []
    
    def ingest_multiple_datasets(
        self, 
        adapters: List[DatasetAdapter],
        **kwargs
    ) -> List[IngestionRecord]:
        """Ingest multiple datasets in sequence."""
        logger.info(f"Starting batch ingestion of {len(adapters)} datasets")
        
        for i, adapter in enumerate(adapters):
            logger.info(f"Processing dataset {i+1}/{len(adapters)}: {adapter.source_name}")
            
            try:
                record = self.pipeline.ingest_dataset(adapter, **kwargs)
                self.results.append(record)
                
                # Brief pause between datasets
                import time
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to process {adapter.source_name}: {e}")
                # Continue with next dataset
                continue
        
        logger.info(f"Batch ingestion completed: {len(self.results)} datasets processed")
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of batch ingestion results."""
        if not self.results:
            return {"error": "No results available"}
        
        total_documents = sum(r.total_documents for r in self.results)
        total_chunks = sum(r.total_chunks for r in self.results)
        successful_chunks = sum(r.successful_chunks for r in self.results)
        failed_chunks = sum(r.failed_chunks for r in self.results)
        
        return {
            "total_datasets": len(self.results),
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "successful_chunks": successful_chunks,
            "failed_chunks": failed_chunks,
            "success_rate": successful_chunks / total_chunks if total_chunks > 0 else 0,
            "datasets": [
                {
                    "name": r.dataset_name,
                    "version": r.dataset_version,
                    "documents": r.total_documents,
                    "chunks": r.total_chunks,
                    "success_rate": r.successful_chunks / r.total_chunks if r.total_chunks > 0 else 0
                }
                for r in self.results
            ]
        }
