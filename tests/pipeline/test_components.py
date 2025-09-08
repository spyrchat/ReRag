"""
Component Integration Tests

Tests for individual pipeline components without requiring local models.
Tests component initialization and configuration validation.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestComponentIntegration:
    """Test pipeline component integration."""
    
    def test_retrieval_component_base_import(self):
        """Test that base retrieval components can be imported."""
        from components.retrieval_pipeline import RetrievalComponent, BaseRetriever, RetrievalResult
        
        assert RetrievalComponent is not None
        assert BaseRetriever is not None
        assert RetrievalResult is not None
    
    def test_retrieval_result_dataclass(self):
        """Test RetrievalResult dataclass functionality."""
        from components.retrieval_pipeline import RetrievalResult
        from langchain_core.documents import Document
        
        doc = Document(page_content="test content", metadata={"test": "value"})
        result = RetrievalResult(
            document=doc,
            score=0.95,
            retrieval_method="dense",
            metadata={"extra": "info"}
        )
        
        assert result.document.page_content == "test content"
        assert result.score == 0.95
        assert result.retrieval_method == "dense"
        assert result.metadata["extra"] == "info"
    
    def test_pipeline_factory_import(self):
        """Test that pipeline factory can be imported."""
        from components.retrieval_pipeline import RetrievalPipelineFactory
        
        assert RetrievalPipelineFactory is not None
        
        # Test that it has required methods
        assert hasattr(RetrievalPipelineFactory, 'create_from_config')
    
    def test_filters_import(self):
        """Test that filter components can be imported."""
        from components.filters import ScoreFilter, ResultLimiter
        
        assert ScoreFilter is not None
        assert ResultLimiter is not None
    
    def test_score_filter_functionality(self):
        """Test score filter without actual retrieval."""
        from components.filters import ScoreFilter
        from components.retrieval_pipeline import RetrievalResult
        from langchain_core.documents import Document
        
        # Create filter
        filter_component = ScoreFilter(min_score=0.5)
        
        # Create test results
        high_score_doc = Document(page_content="high relevance")
        low_score_doc = Document(page_content="low relevance")
        
        results = [
            RetrievalResult(high_score_doc, 0.8, "dense"),
            RetrievalResult(low_score_doc, 0.3, "dense")
        ]
        
        # Filter results
        filtered = filter_component.filter("test query", results)
        
        # Should only keep high score result
        assert len(filtered) == 1
        assert filtered[0].score == 0.8
    
    def test_limit_filter_functionality(self):
        """Test result limiter without actual retrieval."""
        from components.filters import ResultLimiter  
        from components.retrieval_pipeline import RetrievalResult
        from langchain_core.documents import Document
        
        # Create limiter
        limiter = ResultLimiter(max_results=2)
        
        # Create test results
        results = []
        for i in range(5):
            doc = Document(page_content=f"content {i}")
            results.append(RetrievalResult(doc, 0.9 - i*0.1, "dense"))
        
        # Limit results
        limited = limiter.post_process("test query", results)
        
        # Should limit to 2 results
        assert len(limited) == 2
        assert limited[0].score == 0.9
        assert limited[1].score == 0.8
    
    def test_database_controller_import(self):
        """Test that database controllers can be imported."""
        from database.qdrant_controller import QdrantVectorDB
        
        assert QdrantVectorDB is not None
        
        # Test that it has required methods
        db = QdrantVectorDB()
        assert hasattr(db, 'init_collection')
        assert hasattr(db, 'get_client')
        assert hasattr(db, 'insert_documents')
    
    def test_embedding_factory_import(self):
        """Test that embedding factory can be imported."""
        from embedding.factory import get_embedder
        
        assert get_embedder is not None
    
    def test_agent_nodes_import(self):
        """Test that agent nodes can be imported."""
        from agent.nodes.retriever import make_configurable_retriever
        from agent.nodes.generator import make_generator
        from agent.nodes.query_interpreter import make_query_interpreter
        
        assert make_configurable_retriever is not None
        assert make_generator is not None
        assert make_query_interpreter is not None


class TestConfigurationValidation:
    """Test configuration validation without actual initialization."""
    
    def test_config_loader_imports(self):
        """Test config loading utilities."""
        from config.config_loader import load_config, get_retriever_config
        
        assert load_config is not None
        assert get_retriever_config is not None
    
    def test_retrieval_config_structure(self):
        """Test retrieval configuration structure."""
        import yaml
        
        config_path = "pipelines/configs/retrieval/ci_google_gemini.yml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Test required structure
        assert "retrieval_pipeline" in config
        pipeline_config = config["retrieval_pipeline"]
        
        assert "retriever" in pipeline_config
        retriever_config = pipeline_config["retriever"]
        
        # Test retriever config
        assert "type" in retriever_config
        assert "embedding" in retriever_config
        assert "qdrant" in retriever_config
        
        # Test embedding config
        embedding_config = retriever_config["embedding"]
        assert "dense" in embedding_config
        
        dense_config = embedding_config["dense"]
        assert dense_config["provider"] == "google"
        assert "model" in dense_config
        assert "dimensions" in dense_config


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
