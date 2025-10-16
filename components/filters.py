"""
Filter and post-processor components for the modular retrieval pipeline.
"""

from typing import List, Dict, Any, Set
import logging
from components.retrieval_pipeline import ResultFilter, PostProcessor, RetrievalResult
from logs.utils.logger import get_logger

logger = get_logger(__name__)


class ScoreFilter(ResultFilter):
    """Filter results based on minimum score threshold."""
    
    def __init__(self, min_score: float = 0.5):
        self.min_score = min_score
        logger.info(f"Initialized ScoreFilter with min_score={min_score}")
    
    @property
    def component_name(self) -> str:
        return f"score_filter_{self.min_score}"
    
    def filter(self, query: str, results: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        """Filter results below minimum score."""
        min_score = kwargs.get('min_score', self.min_score)
        
        filtered = [r for r in results if r.score >= min_score]
        
        logger.info(f"ScoreFilter: {len(results)} -> {len(filtered)} results (min_score={min_score})")
        return filtered


class MetadataFilter(ResultFilter):
    """Filter results based on metadata criteria."""
    
    def __init__(self, filter_criteria: Dict[str, Any]):
        self.filter_criteria = filter_criteria
        logger.info(f"Initialized MetadataFilter with criteria: {filter_criteria}")
    
    @property
    def component_name(self) -> str:
        return "metadata_filter"
    
    def filter(self, query: str, results: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        """Filter results based on metadata."""
        criteria = kwargs.get('filter_criteria', self.filter_criteria)
        
        filtered = []
        for result in results:
            doc_metadata = result.document.metadata
            labels = doc_metadata.get('labels', {})
            
            # Check each criterion
            passes_filter = True
            for key, expected_value in criteria.items():
                # Check in labels first, then in main metadata
                actual_value = labels.get(key) or doc_metadata.get(key)
                
                if isinstance(expected_value, (list, set)):
                    # Check if actual value is in the expected set
                    if actual_value not in expected_value:
                        passes_filter = False
                        break
                elif isinstance(expected_value, dict):
                    # Handle complex criteria like {"score": {">=": 0.5}}
                    if not self._check_complex_criteria(actual_value, expected_value):
                        passes_filter = False
                        break
                else:
                    # Exact match
                    if actual_value != expected_value:
                        passes_filter = False
                        break
            
            if passes_filter:
                filtered.append(result)
        
        logger.info(f"MetadataFilter: {len(results)} -> {len(filtered)} results")
        return filtered
    
    def _check_complex_criteria(self, value, criteria):
        """Check complex criteria like {">=": 0.5}."""
        for op, threshold in criteria.items():
            if op == ">=":
                return value >= threshold
            elif op == ">":
                return value > threshold
            elif op == "<=":
                return value <= threshold
            elif op == "<":
                return value < threshold
            elif op == "==":
                return value == threshold
            elif op == "!=":
                return value != threshold
            elif op == "in":
                return value in threshold
            elif op == "not_in":
                return value not in threshold
        return True


class TagFilter(ResultFilter):
    """Filter results based on tags."""
    
    def __init__(self, required_tags: List[str] = None, excluded_tags: List[str] = None):
        self.required_tags = set(required_tags or [])
        self.excluded_tags = set(excluded_tags or [])
        logger.info(f"Initialized TagFilter (required: {required_tags}, excluded: {excluded_tags})")
    
    @property
    def component_name(self) -> str:
        return "tag_filter"
    
    def filter(self, query: str, results: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        """Filter results based on tags."""
        required = set(kwargs.get('required_tags', self.required_tags))
        excluded = set(kwargs.get('excluded_tags', self.excluded_tags))
        
        filtered = []
        for result in results:
            labels = result.document.metadata.get('labels', {})
            tags = set(labels.get('tags', []))
            
            # Check required tags
            if required and not required.issubset(tags):
                continue
            
            # Check excluded tags
            if excluded and excluded.intersection(tags):
                continue
            
            filtered.append(result)
        
        logger.info(f"TagFilter: {len(results)} -> {len(filtered)} results")
        return filtered


class DuplicateFilter(ResultFilter):
    """Remove duplicate results based on external_id or content."""
    
    def __init__(self, dedup_by: str = "external_id"):
        self.dedup_by = dedup_by  # "external_id", "content", or "both"
        logger.info(f"Initialized DuplicateFilter (dedup_by={dedup_by})")
    
    @property
    def component_name(self) -> str:
        return f"duplicate_filter_{self.dedup_by}"
    
    def filter(self, query: str, results: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        """Remove duplicates."""
        dedup_by = kwargs.get('dedup_by', self.dedup_by)
        
        seen = set()
        filtered = []
        
        for result in results:
            # Generate deduplication key
            if dedup_by == "external_id":
                key = result.document.metadata.get('external_id')
            elif dedup_by == "content":
                key = result.document.page_content[:200]  # First 200 chars
            elif dedup_by == "both":
                external_id = result.document.metadata.get('external_id', '')
                content_hash = hash(result.document.page_content)
                key = (external_id, content_hash)
            else:
                key = result.document.page_content[:200]
            
            if key not in seen:
                seen.add(key)
                filtered.append(result)
        
        logger.info(f"DuplicateFilter: {len(results)} -> {len(filtered)} results")
        return filtered


class AnswerEnhancer(PostProcessor):
    """Enhance answer results with better formatting and context."""
    
    def __init__(self):
        logger.info("Initialized AnswerEnhancer")
    
    @property
    def component_name(self) -> str:
        return "answer_enhancer"
    
    def post_process(self, query: str, results: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        """Enhance answer formatting and metadata."""
        enhanced = []
        
        for result in results:
            labels = result.document.metadata.get('labels', {})
            
            # Extract answer information
            title = labels.get('title', 'N/A')
            tags = labels.get('tags', [])
            has_context = labels.get('has_question_context', False)
            
            # Enhance metadata
            enhanced_metadata = {
                **result.metadata,
                "question_title": title,
                "programming_tags": tags,
                "has_question_context": has_context,
                "answer_quality": self._assess_answer_quality(result, labels),
                "enhanced": True
            }
            
            # Create enhanced result
            enhanced_result = RetrievalResult(
                document=result.document,
                score=result.score,
                retrieval_method=f"{result.retrieval_method}+enhanced",
                metadata=enhanced_metadata
            )
            enhanced.append(enhanced_result)
        
        logger.info(f"AnswerEnhancer: Enhanced {len(results)} results")
        return enhanced
    
    def _assess_answer_quality(self, result: RetrievalResult, labels: Dict) -> str:
        """Assess the quality of an answer."""
        content_length = len(result.document.page_content)
        has_code = any(marker in result.document.page_content.lower() 
                      for marker in ['```', '<code>', 'def ', 'function', 'class '])
        has_context = labels.get('has_question_context', False)
        tags_count = len(labels.get('tags', []))
        
        score = 0
        if content_length > 200:
            score += 1
        if has_code:
            score += 1
        if has_context:
            score += 1
        if tags_count >= 2:
            score += 1
        
        if score >= 3:
            return "high"
        elif score >= 2:
            return "medium"
        else:
            return "low"


class ContextEnricher(PostProcessor):
    """Enrich results with additional context information."""
    
    def __init__(self):
        logger.info("Initialized ContextEnricher")
    
    @property
    def component_name(self) -> str:
        return "context_enricher"
    
    def post_process(self, query: str, results: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        """Add contextual information to results."""
        enriched = []
        
        for i, result in enumerate(results):
            labels = result.document.metadata.get('labels', {})
            
            # Add positional context
            context_info = {
                "rank_position": i + 1,
                "query_used": query,
                "result_type": "answer" if labels.get('post_type') == 'answer' else "unknown",
                "source_platform": "stackoverflow" if "stackoverflow" in labels.get('source', '') else "unknown",
                "enriched_at": self._get_timestamp()
            }
            
            # Merge with existing metadata
            enriched_metadata = {**result.metadata, **context_info}
            
            enriched_result = RetrievalResult(
                document=result.document,
                score=result.score,
                retrieval_method=f"{result.retrieval_method}+enriched",
                metadata=enriched_metadata
            )
            enriched.append(enriched_result)
        
        logger.info(f"ContextEnricher: Enriched {len(results)} results")
        return enriched
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


class ResultLimiter(PostProcessor):
    """Limit the number of final results."""
    
    def __init__(self, max_results: int = 10):
        self.max_results = max_results
        logger.info(f"Initialized ResultLimiter (max_results={max_results})")
    
    @property
    def component_name(self) -> str:
        return f"result_limiter_{self.max_results}"
    
    def post_process(self, query: str, results: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
        """Limit the number of results."""
        max_results = kwargs.get('max_results', self.max_results)
        limited = results[:max_results]
        
        if len(results) > max_results:
            logger.info(f"ResultLimiter: Limited {len(results)} -> {len(limited)} results")
        
        return limited
