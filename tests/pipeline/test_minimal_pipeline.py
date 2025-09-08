"""
Minimal Pipeline Tests

This module contains minimal tests for the RAG pipeline that:
1. Don't require local model downloads (no sentence transformers)
2. Only use Google Gemini embeddings for CI compatibility
3. Test core pipeline functionality and configuration
4. Validate Qdrant connectivity (optional)

Tests are designed to be runnable in CI/CD environments.
"""

import pytest
import os
import yaml
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class TestMinimalPipeline:
    """Test core pipeline functionality without local models."""
    
    def test_config_loading(self):
        """Test that main config.yml loads correctly."""
        from config.config_loader import load_config
        
        config = load_config("config.yml")
        assert config is not None
        assert isinstance(config, dict)
        
        # Check required sections exist
        assert "embedding" in config
        assert "retrievers" in config
        assert "llm" in config
        
    def test_agent_schema_import(self):
        """Test that agent schema imports correctly."""
        from agent.schema import AgentState
        
        # Test that schema doesn't require SQL fields
        state_annotations = AgentState.__annotations__
        assert "question" in state_annotations
        assert "answer" in state_annotations
        assert "chat_history" in state_annotations
        
    def test_google_embedding_config(self):
        """Test Google embeddings configuration without initialization."""
        from config.config_loader import load_config
        
        config = load_config("config.yml")
        
        # Check Google embeddings are configured in main config
        embedding_config = config.get("embedding", {})
        dense_config = embedding_config.get("dense", {})
        
        assert dense_config.get("provider") == "google"
        assert dense_config.get("model") == "models/embedding-001"
        assert dense_config.get("dimensions") == 768
        assert "api_key_env" in dense_config
        
    def test_ci_google_config_loads(self):
        """Test that CI Google Gemini config loads correctly."""
        ci_config_path = "pipelines/configs/retrieval/ci_google_gemini.yml"
        
        with open(ci_config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        assert config is not None
        retriever_config = config["retrieval_pipeline"]["retriever"]
        
        # Verify Google embeddings
        embedding_config = retriever_config["embedding"]["dense"]
        assert embedding_config["provider"] == "google"
        assert embedding_config["model"] == "models/embedding-001"
        assert embedding_config["dimensions"] == 768
        
    def test_agent_retriever_config_load(self):
        """Test agent retriever can load CI Google config."""
        from bin.agent_retriever import ConfigurableRetrieverAgent
        
        # Use CI Google config
        ci_config_path = "pipelines/configs/retrieval/ci_google_gemini.yml"
        
        # Should not raise exception when loading config
        agent = ConfigurableRetrieverAgent(ci_config_path, cache_pipeline=False)
        config_info = agent.get_config_info()
        
        assert config_info["retriever_type"] == "dense"
        assert "num_stages" in config_info
        
    def test_pipeline_factory_google_config(self):
        """Test pipeline factory with Google config (no retrieval)."""
        from components.retrieval_pipeline import RetrievalPipelineFactory
        
        # Load CI Google config
        ci_config_path = "pipelines/configs/retrieval/ci_google_gemini.yml"
        with open(ci_config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Should be able to parse config without errors
        # Don't actually create pipeline to avoid requiring API keys
        pipeline_config = config["retrieval_pipeline"]
        assert pipeline_config["retriever"]["type"] == "dense"
        assert pipeline_config["retriever"]["embedding"]["dense"]["provider"] == "google"
        
    def test_config_switching(self):
        """Test that configuration switching mechanism works."""
        from config.config_loader import load_config, get_retriever_config
        
        main_config = load_config("config.yml")
        
        # Test extracting different retriever configs
        dense_config = get_retriever_config(main_config, "dense")
        hybrid_config = get_retriever_config(main_config, "hybrid")
        
        assert dense_config["type"] == "dense"
        assert hybrid_config["type"] == "hybrid"
        
        # Both should have Google embeddings
        assert dense_config["embedding"]["provider"] == "google"
        assert hybrid_config["embedding"]["dense"]["provider"] == "google"


class TestConfigValidation:
    """Test configuration file validation."""
    
    def test_yaml_files_valid(self):
        """Test that all YAML files in the project are valid."""
        yaml_files = []
        
        # Find all YAML files
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(('.yml', '.yaml')):
                    yaml_files.append(os.path.join(root, file))
        
        assert len(yaml_files) > 0, "No YAML files found"
        
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r') as f:
                    yaml.safe_load(f)
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML in {yaml_file}: {e}")
                
    def test_google_embeddings_config_complete(self):
        """Test that Google embeddings configurations are complete."""
        config_files = [
            "config.yml",
            "pipelines/configs/retrieval/ci_google_gemini.yml"
        ]
        
        for config_file in config_files:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Find Google embedding configs
            google_configs = self._find_google_configs(config)
            assert len(google_configs) > 0, f"No Google configs found in {config_file}"
            
            for google_config in google_configs:
                assert google_config.get("provider") == "google"
                assert "model" in google_config
                assert "dimensions" in google_config
                assert google_config.get("dimensions") == 768
                
    def _find_google_configs(self, config, path=[]):
        """Recursively find Google embedding configurations."""
        google_configs = []
        
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, dict) and value.get("provider") == "google":
                    google_configs.append(value)
                else:
                    google_configs.extend(self._find_google_configs(value, path + [key]))
        elif isinstance(config, list):
            for i, item in enumerate(config):
                google_configs.extend(self._find_google_configs(item, path + [i]))
                
        return google_configs


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
