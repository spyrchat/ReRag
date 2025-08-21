"""
Reranking components for the modular retrieval pipeline.
"""

from typing import List, Dict, Any
import logging
from components.retrieval_pipeline import Reranker, RetrievalResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker(Reranker):
    """
    Cross-encoder based reranker using sentence-transformers.
    
    Uses models like:
    - ms-marco-MiniLM-L-12-v2
    - ms-marco-MiniLM-L-6-v2
    - cross-encoder/ms-marco-TinyBERT-L-2-v2
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", 
                 device: str = "cpu", top_k: int = None):
        self.model_name = model_name
        self.device = device
        self.top_k = top_k
        self._model = None
        
        logger.info(f"Initialized CrossEncoderReranker with model: {model_name}")
    
    @property
    def component_name(self) -> str:
        return f"cross_encoder_reranker_{self.model_name.split('/')[-1]}"
    
    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name, device=self.device)
                logger.info(f"Loaded CrossEncoder model: {self.model_name}")
            except ImportError:
                raise ImportError("sentence-transformers is required for CrossEncoderReranker")
    
    def rerank(self, query: str, results: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        """Rerank results using cross-encoder model."""
        if not results:
            return results
        
        self._load_model()
        
        # Get top_k from kwargs or use instance default
        top_k = kwargs.get('top_k', self.top_k or len(results))
        
        # Prepare query-document pairs
        query_doc_pairs = []
        for result in results:
            # Use document content for reranking
            doc_text = result.document.page_content
            query_doc_pairs.append([query, doc_text])
        
        # Score with cross-encoder
        try:
            scores = self._model.predict(query_doc_pairs)
            
            # Update results with new scores
            reranked_results = []
            for i, result in enumerate(results):
                new_result = RetrievalResult(
                    document=result.document,
                    score=float(scores[i]),
                    retrieval_method=f"{result.retrieval_method}+cross_encoder",
                    metadata={
                        **result.metadata,
                        "original_score": result.score,
                        "reranker_model": self.model_name,
                        "reranked": True
                    }
                )
                reranked_results.append(new_result)
            
            # Sort by new scores and take top_k
            reranked_results.sort(key=lambda x: x.score, reverse=True)
            final_results = reranked_results[:top_k]
            
            logger.info(f"Reranked {len(results)} results, returning top {len(final_results)}")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in cross-encoder reranking: {e}")
            # Fallback to original results
            return results[:top_k]


class SemanticReranker(Reranker):
    """
    Semantic similarity reranker using embeddings and cosine similarity.
    """
    
    def __init__(self, embedder=None, top_k: int = None):
        self.embedder = embedder
        self.top_k = top_k
        
        logger.info("Initialized SemanticReranker")
    
    @property
    def component_name(self) -> str:
        return "semantic_reranker"
    
    def rerank(self, query: str, results: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        """Rerank using semantic similarity."""
        if not results:
            return results
        
        if not self.embedder:
            logger.warning("No embedder provided, skipping semantic reranking")
            return results
        
        top_k = kwargs.get('top_k', self.top_k or len(results))
        
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Get query embedding
            query_embedding = self.embedder.embed_query(query)
            
            # Get document embeddings
            doc_texts = [result.document.page_content for result in results]
            doc_embeddings = self.embedder.embed_documents(doc_texts)
            
            # Calculate similarities
            similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
            
            # Update results with new scores
            reranked_results = []
            for i, result in enumerate(results):
                new_result = RetrievalResult(
                    document=result.document,
                    score=float(similarities[i]),
                    retrieval_method=f"{result.retrieval_method}+semantic",
                    metadata={
                        **result.metadata,
                        "original_score": result.score,
                        "reranked": True
                    }
                )
                reranked_results.append(new_result)
            
            # Sort and return top_k
            reranked_results.sort(key=lambda x: x.score, reverse=True)
            return reranked_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in semantic reranking: {e}")
            return results[:top_k]


class BM25Reranker(Reranker):
    """
    BM25-based reranker for keyword matching.
    """
    
    def __init__(self, k1: float = 1.2, b: float = 0.75, top_k: int = None):
        self.k1 = k1
        self.b = b
        self.top_k = top_k
        
        logger.info(f"Initialized BM25Reranker (k1={k1}, b={b})")
    
    @property
    def component_name(self) -> str:
        return "bm25_reranker"
    
    def rerank(self, query: str, results: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        """Rerank using BM25 scoring."""
        if not results:
            return results
        
        try:
            from rank_bm25 import BM25Okapi
            import nltk
            from nltk.tokenize import word_tokenize
            
            # Download required NLTK data if not present
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
            top_k = kwargs.get('top_k', self.top_k or len(results))
            
            # Tokenize documents
            doc_texts = [result.document.page_content for result in results]
            tokenized_docs = [word_tokenize(doc.lower()) for doc in doc_texts]
            
            # Create BM25 object
            bm25 = BM25Okapi(tokenized_docs)
            
            # Tokenize query and get scores
            tokenized_query = word_tokenize(query.lower())
            scores = bm25.get_scores(tokenized_query)
            
            # Update results with BM25 scores
            reranked_results = []
            for i, result in enumerate(results):
                new_result = RetrievalResult(
                    document=result.document,
                    score=float(scores[i]),
                    retrieval_method=f"{result.retrieval_method}+bm25",
                    metadata={
                        **result.metadata,
                        "original_score": result.score,
                        "bm25_params": {"k1": self.k1, "b": self.b},
                        "reranked": True
                    }
                )
                reranked_results.append(new_result)
            
            # Sort and return top_k
            reranked_results.sort(key=lambda x: x.score, reverse=True)
            return reranked_results[:top_k]
            
        except ImportError as e:
            logger.error(f"Missing dependency for BM25 reranking: {e}")
            return results[:top_k]
        except Exception as e:
            logger.error(f"Error in BM25 reranking: {e}")
            return results[:top_k]


class EnsembleReranker(Reranker):
    """
    Ensemble reranker that combines multiple reranking strategies.
    """
    
    def __init__(self, rerankers: List[Reranker], weights: List[float] = None, top_k: int = None):
        self.rerankers = rerankers
        self.weights = weights or [1.0] * len(rerankers)
        self.top_k = top_k
        
        if len(self.weights) != len(rerankers):
            raise ValueError("Number of weights must match number of rerankers")
        
        logger.info(f"Initialized EnsembleReranker with {len(rerankers)} rerankers")
    
    @property
    def component_name(self) -> str:
        return f"ensemble_reranker_{len(self.rerankers)}_models"
    
    def rerank(self, query: str, results: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        """Rerank using ensemble of multiple rerankers."""
        if not results:
            return results
        
        top_k = kwargs.get('top_k', self.top_k or len(results))
        
        # Get scores from each reranker
        all_scores = []
        for reranker in self.rerankers:
            try:
                reranked = reranker.rerank(query, results, **kwargs)
                # Extract scores in same order as input
                scores = []
                for orig_result in results:
                    # Find corresponding result in reranked list
                    orig_id = id(orig_result.document)
                    for reranked_result in reranked:
                        if id(reranked_result.document) == orig_id:
                            scores.append(reranked_result.score)
                            break
                    else:
                        scores.append(0.0)  # Not found
                all_scores.append(scores)
            except Exception as e:
                logger.error(f"Error in {reranker.component_name}: {e}")
                # Use zero scores for failed reranker
                all_scores.append([0.0] * len(results))
        
        # Normalize scores and compute weighted average
        import numpy as np
        
        normalized_scores = []
        for scores in all_scores:
            scores_array = np.array(scores)
            if scores_array.max() > scores_array.min():
                # Min-max normalization
                normalized = (scores_array - scores_array.min()) / (scores_array.max() - scores_array.min())
            else:
                normalized = scores_array
            normalized_scores.append(normalized)
        
        # Weighted combination
        ensemble_scores = np.zeros(len(results))
        for i, (scores, weight) in enumerate(zip(normalized_scores, self.weights)):
            ensemble_scores += weight * scores
        
        # Create final results
        final_results = []
        for i, result in enumerate(results):
            new_result = RetrievalResult(
                document=result.document,
                score=float(ensemble_scores[i]),
                retrieval_method=f"{result.retrieval_method}+ensemble",
                metadata={
                    **result.metadata,
                    "original_score": result.score,
                    "ensemble_components": [r.component_name for r in self.rerankers],
                    "ensemble_weights": self.weights,
                    "reranked": True
                }
            )
            final_results.append(new_result)
        
        # Sort and return top_k
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:top_k]
