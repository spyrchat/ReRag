"""
Enhanced embedding pipeline with strategy selection and batch processing.
Supports dense, sparse, and hybrid embeddings with caching and error handling.
"""
import os
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import logging
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from embedding.factory import get_embedder
from pipelines.contracts import ChunkMeta, build_doc_id, build_chunk_id, compute_content_hash, DatasetSplit


logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    """Enhanced embedding pipeline with strategy selection and caching."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_strategy = config.get("embedding_strategy", "dense").lower()
        
        # Initialize embedders based on strategy
        self.dense_embedder = None
        self.sparse_embedder = None
        
        if self.embedding_strategy in ["dense", "hybrid"]:
            dense_config = config.get("embedding", {}).get("dense", {})
            self.dense_embedder = get_embedder(dense_config)
            logger.info(f"Initialized dense embedder: {dense_config.get('provider', 'unknown')}")
        
        if self.embedding_strategy in ["sparse", "hybrid"]:
            sparse_config = config.get("embedding", {}).get("sparse", {})
            self.sparse_embedder = get_embedder(sparse_config)
            logger.info(f"Initialized sparse embedder: {sparse_config.get('provider', 'unknown')}")
        
        # Caching configuration
        self.enable_cache = config.get("embedding_cache", {}).get("enabled", True)
        self.cache_dir = Path(config.get("embedding_cache", {}).get("dir", "cache/embeddings"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Batch processing
        self.batch_size = config.get("embedding", {}).get("batch_size", 32)
        self.max_retries = config.get("embedding", {}).get("max_retries", 3)
        
        # Error handling
        self.fail_fast = config.get("embedding", {}).get("fail_fast", False)
        self.fallback_embedding_dim = config.get("embedding", {}).get("fallback_dim", 384)
    
    def process_documents(self, documents: List[Document]) -> List[ChunkMeta]:
        """Process documents into ChunkMeta with embeddings."""
        if not documents:
            return []
        
        logger.info(f"Processing {len(documents)} documents with {self.embedding_strategy} strategy")
        
        chunk_metas = []
        
        # Convert documents to ChunkMeta objects
        for doc in documents:
            chunk_meta = self._document_to_chunk_meta(doc)
            chunk_metas.append(chunk_meta)
        
        # Generate embeddings in batches
        if self.dense_embedder:
            logger.info("Generating dense embeddings...")
            dense_embeddings = self._generate_embeddings(
                [meta.text for meta in chunk_metas], 
                self.dense_embedder, 
                "dense"
            )
            
            for meta, embedding in zip(chunk_metas, dense_embeddings):
                meta.dense_embedding = embedding
                if embedding:
                    meta.embedding_dim = len(embedding)
                    meta.embedding_model = self._get_model_name(self.dense_embedder)
        
        if self.sparse_embedder:
            logger.info("Generating sparse embeddings...")
            sparse_embeddings = self._generate_embeddings(
                [meta.text for meta in chunk_metas], 
                self.sparse_embedder, 
                "sparse"
            )
            
            for meta, embedding in zip(chunk_metas, sparse_embeddings):
                meta.sparse_embedding = embedding
        
        logger.info(f"Successfully processed {len(chunk_metas)} chunk metas")
        return chunk_metas
    
    def _document_to_chunk_meta(self, doc: Document) -> ChunkMeta:
        """Convert Document to ChunkMeta with deterministic IDs."""
        text = doc.page_content
        metadata = doc.metadata
        
        # Generate deterministic IDs
        doc_sha256 = compute_content_hash(text)
        source = metadata.get("source", "unknown")
        external_id = metadata.get("external_id", "unknown")
        
        doc_id = build_doc_id(source, external_id, doc_sha256)
        chunk_index = metadata.get("chunk_index", 0)
        chunk_id = build_chunk_id(doc_id, chunk_index)
        
        # Extract git commit and config hash from environment/config
        git_commit = self._get_git_commit()
        config_hash = self._compute_config_hash()
        
        chunk_meta = ChunkMeta(
            doc_id=doc_id,
            chunk_id=chunk_id,
            doc_sha256=doc_sha256,
            text=text,
            source=source,
            dataset_version=metadata.get("dataset_version", "unknown"),
            external_id=external_id,
            uri=metadata.get("uri"),
            chunk_index=chunk_index,
            num_chunks=metadata.get("num_chunks", 1),
            token_count=metadata.get("token_estimate"),
            char_count=len(text),
            split=DatasetSplit(metadata.get("split", "all")),
            labels=metadata.get("labels", {}),
            git_commit=git_commit,
            config_hash=config_hash
        )
        
        # Copy additional metadata
        for key, value in metadata.items():
            if key not in chunk_meta.dict():
                chunk_meta.labels[key] = value
        
        return chunk_meta
    
    def _generate_embeddings(self, texts: List[str], embedder: Embeddings, embedding_type: str) -> List[Optional[Union[List[float], Dict[int, float]]]]:
        """Generate embeddings with caching and error handling."""
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self._process_embedding_batch(batch_texts, embedder, embedding_type)
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def _process_embedding_batch(self, texts: List[str], embedder: Embeddings, embedding_type: str) -> List[Optional[Union[List[float], Dict[int, float]]]]:
        """Process a batch of texts for embedding."""
        batch_embeddings = []
        
        for text in texts:
            try:
                # Check cache first
                if self.enable_cache:
                    cached_embedding = self._get_cached_embedding(text, embedder, embedding_type)
                    if cached_embedding is not None:
                        batch_embeddings.append(cached_embedding)
                        continue
                
                # Generate new embedding
                embedding = self._generate_single_embedding(text, embedder, embedding_type)
                
                # Cache the result
                if self.enable_cache and embedding is not None:
                    self._cache_embedding(text, embedding, embedder, embedding_type)
                
                batch_embeddings.append(embedding)
                
            except Exception as e:
                logger.error(f"Error generating {embedding_type} embedding for text: {text[:100]}... Error: {e}")
                
                if self.fail_fast:
                    raise
                
                # Add fallback embedding
                fallback_embedding = self._create_fallback_embedding(embedding_type)
                batch_embeddings.append(fallback_embedding)
        
        return batch_embeddings
    
    def _generate_single_embedding(self, text: str, embedder: Embeddings, embedding_type: str) -> Optional[Union[List[float], Dict[int, float]]]:
        """Generate embedding for a single text with retries."""
        for attempt in range(self.max_retries):
            try:
                embedding = embedder.embed_query(text)
                
                # Validate embedding based on type
                if embedding_type == "dense":
                    if embedding and isinstance(embedding, list) and len(embedding) > 0:
                        return embedding
                else:  # sparse
                    # Handle different sparse embedding formats
                    if embedding is not None:
                        if isinstance(embedding, dict) and len(embedding) > 0:
                            return embedding
                        elif hasattr(embedding, '__len__') and len(embedding) > 0:
                            # Convert array-like objects to dict if needed
                            logger.warning(f"Converting sparse embedding type {type(embedding)} to dict")
                            return dict(embedding) if hasattr(embedding, 'items') else {}
                        else:
                            logger.warning(f"Unexpected sparse embedding type: {type(embedding)}")
                            return {}
                
                logger.warning(f"Empty or invalid {embedding_type} embedding returned for text: {text[:50]}...")
                return None
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed for {embedding_type} embedding: {e}")
                if attempt == self.max_retries - 1:
                    raise
        
        return None
    
    def _get_cached_embedding(self, text: str, embedder: Embeddings, embedding_type: str) -> Optional[Union[List[float], Dict[int, float]]]:
        """Retrieve cached embedding."""
        cache_key = self._compute_cache_key(text, embedder, embedding_type)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    embedding = data.get("embedding")
                    
                    # Convert string keys back to int for sparse embeddings
                    if embedding_type == "sparse" and isinstance(embedding, dict):
                        return {int(k): v for k, v in embedding.items()}
                    
                    return embedding
            except Exception as e:
                logger.warning(f"Error reading cache file {cache_file}: {e}")
        
        return None
    
    def _cache_embedding(self, text: str, embedding: Union[List[float], Dict[int, float]], embedder: Embeddings, embedding_type: str):
        """Cache embedding result."""
        cache_key = self._compute_cache_key(text, embedder, embedding_type)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            # Convert int keys to string for JSON serialization
            serializable_embedding = embedding
            if embedding_type == "sparse" and isinstance(embedding, dict):
                serializable_embedding = {str(k): v for k, v in embedding.items()}
            
            cache_data = {
                "text_hash": hashlib.sha256(text.encode()).hexdigest(),
                "embedding_type": embedding_type,
                "model": self._get_model_name(embedder),
                "embedding": serializable_embedding,
                "created_at": str(datetime.now())
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
                
        except Exception as e:
            logger.warning(f"Error caching embedding: {e}")
    
    def _compute_cache_key(self, text: str, embedder: Embeddings, embedding_type: str) -> str:
        """Compute cache key for text and embedder."""
        model_name = self._get_model_name(embedder)
        content = f"{text}:{embedding_type}:{model_name}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _get_model_name(self, embedder: Embeddings) -> str:
        """Extract model name from embedder."""
        if hasattr(embedder, 'model_name'):
            return embedder.model_name
        elif hasattr(embedder, 'model'):
            return embedder.model
        else:
            return embedder.__class__.__name__
    
    def _create_fallback_embedding(self, embedding_type: str) -> Union[List[float], Dict[int, float]]:
        """Create fallback embedding for failed cases."""
        if embedding_type == "sparse":
            # For sparse embeddings, return empty dict
            return {}
        else:
            # For dense embeddings, return zero vector
            return [0.0] * self.fallback_embedding_dim
    
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
    
    def _compute_config_hash(self) -> str:
        """Compute hash of current configuration."""
        config_str = json.dumps(self.config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]
