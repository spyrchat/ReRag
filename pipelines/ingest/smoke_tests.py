"""
Post-ingestion smoke tests to verify system quality.
Implements "trust but verify" principle with golden queries.
"""
import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from pipelines.contracts import SmokeTestResult
from database.qdrant_controller import QdrantVectorDB
from embedding.factory import get_embedder


logger = logging.getLogger(__name__)


class SmokeTest(ABC):
    """Abstract base class for smoke tests."""
    
    @property
    @abstractmethod
    def test_name(self) -> str:
        """Return the name of this test."""
        pass
    
    @abstractmethod
    def run(self, config: Dict[str, Any]) -> SmokeTestResult:
        """Run the smoke test."""
        pass


class CollectionExistsTest(SmokeTest):
    """Test that the collection exists and has expected properties."""
    
    @property
    def test_name(self) -> str:
        return "collection_exists"
    
    def run(self, config: Dict[str, Any]) -> SmokeTestResult:
        """Check if collection exists and is properly configured."""
        try:
            vector_db = QdrantVectorDB()
            client = vector_db.get_client()
            
            # Use actual collection name if provided, otherwise fall back to configured name
            collection_name = config.get("actual_collection_name") or vector_db.get_collection_name()
            
            # Check if collection exists
            if not client.collection_exists(collection_name):
                return SmokeTestResult(
                    passed=False,
                    test_name=self.test_name,
                    errors=[f"Collection {collection_name} does not exist"]
                )
            
            # Get collection info
            info = client.get_collection(collection_name)
            
            details = {
                "collection_name": collection_name,
                "points_count": info.points_count,
                "status": info.status,
                "vectors_config": len(info.config.params.vectors or {}),
                "sparse_vectors_config": len(info.config.params.sparse_vectors or {})
            }
            
            # Basic validation
            errors = []
            if info.points_count == 0:
                errors.append("Collection is empty")
            
            return SmokeTestResult(
                passed=len(errors) == 0,
                test_name=self.test_name,
                details=details,
                errors=errors
            )
            
        except Exception as e:
            return SmokeTestResult(
                passed=False,
                test_name=self.test_name,
                errors=[f"Error checking collection: {str(e)}"]
            )


class VectorDimensionTest(SmokeTest):
    """Test that vectors have expected dimensions."""
    
    @property
    def test_name(self) -> str:
        return "vector_dimensions"
    
    def run(self, config: Dict[str, Any]) -> SmokeTestResult:
        """Check vector dimensions match expectations."""
        try:
            vector_db = QdrantVectorDB()
            client = vector_db.get_client()
            
            # Use actual collection name if provided, otherwise fall back to configured name
            collection_name = config.get("actual_collection_name") or vector_db.get_collection_name()
            
            # Sample a few points
            sample_points = client.scroll(
                collection_name=collection_name,
                limit=5
            )[0]
            
            if not sample_points:
                return SmokeTestResult(
                    passed=False,
                    test_name=self.test_name,
                    errors=["No points found in collection"]
                )
            
            # Check dimensions
            details = {}
            errors = []
            
            for point in sample_points:
                if hasattr(point, 'vector') and point.vector:
                    for vector_name, vector_data in point.vector.items():
                        if vector_name not in details:
                            details[vector_name] = {
                                "dimensions": [],
                                "sample_count": 0
                            }
                        
                        if hasattr(vector_data, '__len__'):
                            details[vector_name]["dimensions"].append(len(vector_data))
                            details[vector_name]["sample_count"] += 1
            
            # Validate consistency
            for vector_name, info in details.items():
                dims = info["dimensions"]
                if len(set(dims)) > 1:
                    errors.append(f"Inconsistent dimensions for {vector_name}: {set(dims)}")
                elif dims:
                    info["consistent_dimension"] = dims[0]
            
            return SmokeTestResult(
                passed=len(errors) == 0,
                test_name=self.test_name,
                details=details,
                errors=errors
            )
            
        except Exception as e:
            return SmokeTestResult(
                passed=False,
                test_name=self.test_name,
                errors=[f"Error checking vector dimensions: {str(e)}"]
            )


class GoldenQueryTest(SmokeTest):
    """Test retrieval with known golden queries."""
    
    def __init__(self, golden_queries: List[Dict[str, Any]]):
        self.golden_queries = golden_queries
    
    @property
    def test_name(self) -> str:
        return "golden_queries"
    
    def run(self, config: Dict[str, Any]) -> SmokeTestResult:
        """Run golden queries and check recall."""
        try:
            from retrievers.router import RetrieverRouter
            
            # Initialize retriever
            retriever_router = RetrieverRouter(config)
            
            results = []
            total_queries = len(self.golden_queries)
            successful_queries = 0
            
            for query_info in self.golden_queries:
                query = query_info["query"]
                expected_doc_ids = set(query_info.get("relevant_doc_ids", []))
                min_recall = query_info.get("min_recall", 0.1)
                
                try:
                    # Retrieve documents
                    retrieved_docs = retriever_router.retrieve(query)
                    
                    # Extract document IDs
                    if hasattr(retrieved_docs[0], 'metadata'):
                        retrieved_ids = set(doc.metadata.get("external_id") for doc in retrieved_docs)
                    else:
                        # Handle tuple format (doc, score)
                        retrieved_ids = set(doc.metadata.get("external_id") for doc, _ in retrieved_docs)
                    
                    # Calculate recall
                    if expected_doc_ids:
                        overlap = expected_doc_ids.intersection(retrieved_ids)
                        recall = len(overlap) / len(expected_doc_ids)
                    else:
                        # If no expected docs, just check that we got results
                        recall = 1.0 if retrieved_ids else 0.0
                    
                    query_result = {
                        "query": query,
                        "recall": recall,
                        "retrieved_count": len(retrieved_ids),
                        "expected_count": len(expected_doc_ids),
                        "overlap_count": len(overlap) if expected_doc_ids else 0,
                        "passed": recall >= min_recall
                    }
                    
                    results.append(query_result)
                    
                    if recall >= min_recall:
                        successful_queries += 1
                        
                except Exception as e:
                    logger.error(f"Error running golden query '{query}': {e}")
                    results.append({
                        "query": query,
                        "error": str(e),
                        "passed": False
                    })
            
            # Calculate overall success rate
            success_rate = successful_queries / total_queries if total_queries > 0 else 0
            min_success_rate = config.get("smoke_tests", {}).get("min_success_rate", 0.8)
            
            details = {
                "total_queries": total_queries,
                "successful_queries": successful_queries,
                "success_rate": success_rate,
                "min_success_rate": min_success_rate,
                "query_results": results
            }
            
            errors = []
            if success_rate < min_success_rate:
                errors.append(f"Success rate {success_rate:.2%} below threshold {min_success_rate:.2%}")
            
            return SmokeTestResult(
                passed=len(errors) == 0,
                test_name=self.test_name,
                details=details,
                errors=errors
            )
            
        except Exception as e:
            return SmokeTestResult(
                passed=False,
                test_name=self.test_name,
                errors=[f"Error running golden queries: {str(e)}"]
            )


class EmbeddingQualityTest(SmokeTest):
    """Test embedding quality with sanity checks."""
    
    @property
    def test_name(self) -> str:
        return "embedding_quality"
    
    def run(self, config: Dict[str, Any]) -> SmokeTestResult:
        """Check embedding quality metrics."""
        try:
            vector_db = QdrantVectorDB()
            client = vector_db.get_client()
            
            # Use actual collection name if provided, otherwise fall back to configured name
            collection_name = config.get("actual_collection_name") or vector_db.get_collection_name()
            
            # Sample points for analysis
            sample_points = client.scroll(
                collection_name=collection_name,
                limit=100
            )[0]
            
            if not sample_points:
                return SmokeTestResult(
                    passed=False,
                    test_name=self.test_name,
                    errors=["No points found for quality analysis"]
                )
            
            details = {}
            errors = []
            warnings = []
            
            # Analyze each vector type
            for point in sample_points:
                if hasattr(point, 'vector') and point.vector:
                    for vector_name, vector_data in point.vector.items():
                        if vector_name not in details:
                            details[vector_name] = {
                                "sample_count": 0,
                                "zero_vectors": 0,
                                "norms": [],
                                "dimension": None
                            }
                        
                        analysis = details[vector_name]
                        analysis["sample_count"] += 1
                        
                        # Check for zero vectors
                        if hasattr(vector_data, '__iter__'):
                            if analysis["dimension"] is None:
                                analysis["dimension"] = len(vector_data)
                            
                            # Calculate norm
                            norm = sum(x*x for x in vector_data) ** 0.5
                            analysis["norms"].append(norm)
                            
                            if norm < 1e-6:  # Essentially zero
                                analysis["zero_vectors"] += 1
            
            # Quality checks
            for vector_name, analysis in details.items():
                sample_count = analysis["sample_count"]
                zero_count = analysis["zero_vectors"]
                norms = analysis["norms"]
                
                # Check for excessive zero vectors
                zero_ratio = zero_count / sample_count if sample_count > 0 else 0
                if zero_ratio > 0.1:  # More than 10% zero vectors
                    errors.append(f"{vector_name}: High zero vector ratio: {zero_ratio:.2%}")
                
                # Check norm distribution
                if norms:
                    avg_norm = sum(norms) / len(norms)
                    analysis["avg_norm"] = avg_norm
                    analysis["min_norm"] = min(norms)
                    analysis["max_norm"] = max(norms)
                    
                    # Sanity check norms
                    if avg_norm < 0.1:
                        warnings.append(f"{vector_name}: Very low average norm: {avg_norm:.4f}")
                    elif avg_norm > 100:
                        warnings.append(f"{vector_name}: Very high average norm: {avg_norm:.4f}")
            
            return SmokeTestResult(
                passed=len(errors) == 0,
                test_name=self.test_name,
                details=details,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            return SmokeTestResult(
                passed=False,
                test_name=self.test_name,
                errors=[f"Error analyzing embedding quality: {str(e)}"]
            )


class SmokeTestRunner:
    """Runs all smoke tests and aggregates results."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tests = self._initialize_tests()
    
    def _initialize_tests(self) -> List[SmokeTest]:
        """Initialize the standard set of smoke tests."""
        tests = [
            CollectionExistsTest(),
            VectorDimensionTest(),
            EmbeddingQualityTest()
        ]
        
        # Add golden query test if queries provided
        golden_queries = self.config.get("smoke_tests", {}).get("golden_queries", [])
        if golden_queries:
            tests.append(GoldenQueryTest(golden_queries))
        
        return tests
    
    def run_all_tests(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Run all smoke tests and return aggregated results."""
        logger.info(f"Running {len(self.tests)} smoke tests...")
        
        # Update config with actual collection name if provided
        test_config = self.config.copy()
        if collection_name:
            test_config["actual_collection_name"] = collection_name
        
        results = []
        passed_count = 0
        
        for test in self.tests:
            logger.info(f"Running test: {test.test_name}")
            result = test.run(test_config)
            results.append(result)
            
            if result.passed:
                passed_count += 1
                logger.info(f"✓ {test.test_name} passed")
            else:
                logger.error(f"✗ {test.test_name} failed: {result.errors}")
        
        # Aggregate results
        total_tests = len(self.tests)
        success_rate = passed_count / total_tests if total_tests > 0 else 0
        overall_passed = success_rate >= self.config.get("smoke_tests", {}).get("min_overall_success", 0.8)
        
        summary = {
            "overall_passed": overall_passed,
            "success_rate": success_rate,
            "total_tests": total_tests,
            "passed_tests": passed_count,
            "failed_tests": total_tests - passed_count,
            "test_results": [result.dict() for result in results]
        }
        
        if overall_passed:
            logger.info(f"✓ All smoke tests passed ({success_rate:.1%} success rate)")
        else:
            logger.error(f"✗ Smoke tests failed ({success_rate:.1%} success rate)")
        
        return summary
