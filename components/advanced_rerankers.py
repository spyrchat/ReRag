"""
Advanced reranking components for the modular retrieval pipeline.
Demonstrates how easy it is to add new rerankers.
"""

from typing import List, Dict, Any
import logging
import numpy as np
from components.retrieval_pipeline import Reranker, RetrievalResult

logger = logging.getLogger(__name__)


class CohereBReranker(Reranker):
    """
    Cohere Rerank API-based reranker.
    High-quality reranking using Cohere's commercial models.
    """
    
    def __init__(self, api_key: str = None, model: str = "rerank-english-v2.0", 
                 top_k: int = None):
        self.api_key = api_key
        self.model = model
        self.top_k = top_k
        self._client = None
        
        logger.info(f"Initialized CohereBReranker with model: {model}")
    
    @property
    def component_name(self) -> str:
        return f"cohere_reranker_{self.model.replace('-', '_')}"
    
    def _load_client(self):
        """Lazy load the Cohere client."""
        if self._client is None:
            try:
                import cohere
                self._client = cohere.Client(self.api_key)
                logger.info(f"Loaded Cohere client with model: {self.model}")
            except ImportError:
                raise ImportError("cohere package is required for CohereBReranker")
            except Exception as e:
                logger.warning(f"Could not initialize Cohere client: {e}")
                raise
    
    def rerank(self, query: str, results: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        """Rerank results using Cohere Rerank API."""
        if not results:
            return results
        
        try:
            self._load_client()
            
            # Prepare documents for reranking
            documents = [result.document.page_content for result in results]
            
            # Call Cohere Rerank API
            response = self._client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_k=self.top_k or len(results)
            )
            
            # Reorder results based on Cohere scores
            reranked_results = []
            for rank_result in response.results:
                original_result = results[rank_result.index]
                
                # Store original score in metadata
                original_result.metadata["original_score"] = original_result.score
                original_result.metadata["cohere_score"] = rank_result.relevance_score
                
                # Update score and method
                original_result.score = rank_result.relevance_score
                original_result.retrieval_method = f"{original_result.retrieval_method}+cohere"
                
                reranked_results.append(original_result)
            
            logger.info(f"Reranked {len(results)} results with Cohere, returning top {len(reranked_results)}")
            return reranked_results
            
        except Exception as e:
            logger.warning(f"Cohere reranking failed: {e}, returning original results")
            return results


class BgeReranker(Reranker):
    """
    BGE (BAAI General Embedding) reranker using local models.
    Excellent for multilingual and general domain reranking.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", 
                 device: str = "cpu", top_k: int = None):
        self.model_name = model_name
        self.device = device
        self.top_k = top_k
        self._tokenizer = None
        self._model = None
        
        logger.info(f"Initialized BgeReranker with model: {model_name}")
    
    @property
    def component_name(self) -> str:
        return f"bge_reranker_{self.model_name.split('/')[-1]}"
    
    def _load_model(self):
        """Lazy load the BGE model."""
        if self._model is None:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                
                if self.device == "cuda" and torch.cuda.is_available():
                    self._model = self._model.cuda()
                
                logger.info(f"Loaded BGE model: {self.model_name}")
            except ImportError:
                raise ImportError("transformers and torch are required for BgeReranker")
    
    def rerank(self, query: str, results: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        """Rerank results using BGE model."""
        if not results:
            return results
        
        self._load_model()
        
        try:
            import torch
            
            # Prepare query-document pairs
            pairs = [[query, result.document.page_content] for result in results]
            
            # Tokenize and compute scores
            with torch.no_grad():
                inputs = self._tokenizer(pairs, padding=True, truncation=True, 
                                       return_tensors='pt', max_length=512)
                
                if self.device == "cuda" and torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                scores = self._model(**inputs, return_dict=True).logits.view(-1, ).float()
                scores = torch.sigmoid(scores).cpu().numpy()
            
            # Create scored results
            scored_results = []
            for i, result in enumerate(results):
                # Store original score
                result.metadata["original_score"] = result.score
                result.metadata["bge_score"] = float(scores[i])
                
                # Update score and method
                result.score = float(scores[i])
                result.retrieval_method = f"{result.retrieval_method}+bge"
                
                scored_results.append((result, float(scores[i])))
            
            # Sort by BGE score
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # Return top k
            top_k = self.top_k or len(scored_results)
            reranked_results = [result for result, _ in scored_results[:top_k]]
            
            logger.info(f"Reranked {len(results)} results with BGE, returning top {len(reranked_results)}")
            return reranked_results
            
        except Exception as e:
            logger.warning(f"BGE reranking failed: {e}, returning original results")
            return results


class ColBERTReranker(Reranker):
    """
    ColBERT-based reranker for late interaction reranking.
    Highly effective for passage ranking tasks.
    """
    
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0", 
                 device: str = "cpu", top_k: int = None):
        self.model_name = model_name
        self.device = device
        self.top_k = top_k
        self._model = None
        
        logger.info(f"Initialized ColBERTReranker with model: {model_name}")
    
    @property
    def component_name(self) -> str:
        return f"colbert_reranker_{self.model_name.split('/')[-1]}"
    
    def _load_model(self):
        """Lazy load the ColBERT model."""
        if self._model is None:
            try:
                from colbert.modeling.checkpoint import Checkpoint
                
                self._model = Checkpoint(self.model_name, colbert_config=None)
                logger.info(f"Loaded ColBERT model: {self.model_name}")
            except ImportError:
                raise ImportError("colbert-ai package is required for ColBERTReranker")
    
    def rerank(self, query: str, results: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        """Rerank results using ColBERT late interaction."""
        if not results:
            return results
        
        try:
            self._load_model()
            
            # Prepare documents
            documents = [result.document.page_content for result in results]
            
            # Compute ColBERT scores
            Q = self._model.queryFromText([query])
            D = self._model.docFromText(documents)
            
            # Late interaction scoring
            scores = self._model.score(Q, D).squeeze().tolist()
            if isinstance(scores, float):  # Single document
                scores = [scores]
            
            # Create scored results
            scored_results = []
            for i, result in enumerate(results):
                # Store original score
                result.metadata["original_score"] = result.score
                result.metadata["colbert_score"] = scores[i]
                
                # Update score and method
                result.score = scores[i]
                result.retrieval_method = f"{result.retrieval_method}+colbert"
                
                scored_results.append((result, scores[i]))
            
            # Sort by ColBERT score
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # Return top k
            top_k = self.top_k or len(scored_results)
            reranked_results = [result for result, _ in scored_results[:top_k]]
            
            logger.info(f"Reranked {len(results)} results with ColBERT, returning top {len(reranked_results)}")
            return reranked_results
            
        except Exception as e:
            logger.warning(f"ColBERT reranking failed: {e}, returning original results")
            return results


class MultiStageReranker(Reranker):
    """
    Multi-stage reranker that combines multiple reranking models.
    First stage: Fast lightweight reranker (e.g., BGE-small)
    Second stage: High-quality reranker (e.g., Cohere, CrossEncoder-large)
    """
    
    def __init__(self, stage1_reranker: Reranker, stage2_reranker: Reranker,
                 stage1_k: int = 20, stage2_k: int = None):
        self.stage1_reranker = stage1_reranker
        self.stage2_reranker = stage2_reranker
        self.stage1_k = stage1_k
        self.stage2_k = stage2_k
        
        logger.info(f"Initialized MultiStageReranker: {stage1_reranker.component_name} -> {stage2_reranker.component_name}")
    
    @property
    def component_name(self) -> str:
        return f"multistage_{self.stage1_reranker.component_name}_{self.stage2_reranker.component_name}"
    
    def rerank(self, query: str, results: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        """Apply two-stage reranking."""
        if not results:
            return results
        
        logger.info(f"Multi-stage reranking: Stage 1 with {self.stage1_reranker.component_name}")
        
        # Stage 1: Fast reranking to reduce candidate set
        stage1_results = self.stage1_reranker.rerank(query, results, **kwargs)
        stage1_results = stage1_results[:self.stage1_k]
        
        logger.info(f"Multi-stage reranking: Stage 2 with {self.stage2_reranker.component_name}")
        
        # Stage 2: High-quality reranking on reduced set
        stage2_results = self.stage2_reranker.rerank(query, stage1_results, **kwargs)
        
        if self.stage2_k:
            stage2_results = stage2_results[:self.stage2_k]
        
        # Update retrieval method to reflect multi-stage process
        for result in stage2_results:
            result.retrieval_method = f"{result.retrieval_method}+multistage"
            result.metadata["multistage_reranking"] = True
        
        logger.info(f"Multi-stage reranking completed: {len(results)} -> {self.stage1_k} -> {len(stage2_results)}")
        return stage2_results
