#!/usr/bin/env python3
"""
Test script to verify sparse embedding serialization fixes.
"""
import sys
import os
sys.path.append('/home/spiros/Desktop/Thesis/Thesis')

import yaml
import logging
from pathlib import Path
from langchain_core.documents import Document

from pipelines.ingest.embedder import EmbeddingPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_sparse_embeddings():
    """Test sparse embedding generation and serialization."""
    
    # Load config
    config_path = Path("/home/spiros/Desktop/Thesis/Thesis/pipelines/configs/stackoverflow_hybrid.yml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create test documents
    test_docs = [
        Document(
            page_content="Python is a programming language used for machine learning and data science.",
            metadata={
                "source": "test",
                "external_id": "test_1",
                "chunk_index": 0,
                "num_chunks": 1,
                "split": "test"
            }
        ),
        Document(
            page_content="Stack Overflow is a question and answer site for programmers.",
            metadata={
                "source": "test", 
                "external_id": "test_2",
                "chunk_index": 0,
                "num_chunks": 1,
                "split": "test"
            }
        )
    ]
    
    try:
        # Initialize embedding pipeline
        logger.info("Initializing embedding pipeline with hybrid strategy...")
        embedder = EmbeddingPipeline(config)
        
        # Process documents
        logger.info("Processing test documents...")
        chunk_metas = embedder.process_documents(test_docs)
        
        # Check results
        logger.info(f"Generated {len(chunk_metas)} chunk metas")
        
        for i, meta in enumerate(chunk_metas):
            logger.info(f"\nChunk {i+1}:")
            logger.info(f"  Text: {meta.text[:50]}...")
            logger.info(f"  Dense embedding: {type(meta.dense_embedding)} dim={meta.embedding_dim}")
            logger.info(f"  Sparse embedding: {type(meta.sparse_embedding)}")
            
            if meta.sparse_embedding:
                logger.info(f"  Sparse size: {len(meta.sparse_embedding)} tokens")
                # Show first few sparse entries
                sparse_items = list(meta.sparse_embedding.items())[:5]
                logger.info(f"  Sparse sample: {sparse_items}")
        
        logger.info("✅ Sparse embedding test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_sparse_embeddings()
    sys.exit(0 if success else 1)
