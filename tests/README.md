# Testing Framework

**Last Verified:** 2025-10-08  
**Status:** ✅ Verified against actual codebase

Comprehensive testing suite ensuring system reliability and correctness across all components.

---

## 📁 Test Structure

```
tests/
├── README.md
├── requirements-minimal.txt
├── __init__.py
└── pipeline/
    ├── __init__.py
    ├── run_tests.py
    ├── test_runner.py
    ├── test_minimal.py
    ├── test_config.py
    ├── test_qdrant.py
    ├── test_components.py
    ├── test_minimal_pipeline.py
    ├── test_qdrant_connectivity.py
    └── test_end_to_end.py
```

**Total:** 9 test files, ~41 test methods

---

## 🚀 Running Tests

### Quick Test Run
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/pipeline/test_minimal.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html

# Custom test runner
python tests/pipeline/run_tests.py
```

---

## 📋 Real Test Files

### 1. test_minimal.py (6 functions)
- test_config_loading()
- test_agent_schema()
- test_google_embeddings_config()
- test_agent_retriever_with_google()
- test_pipeline_factory_google_only()
- test_config_switching()

### 2. test_config.py (4 functions)
- test_yaml_validity()
- test_retrieval_config_structure()
- test_google_embeddings_in_configs()
- test_main_config_structure()

### 3. test_qdrant.py (3 functions)
- test_qdrant_connectivity()
- test_qdrant_collections_endpoint()
- test_create_delete_collection()

### 4. test_components.py (2 classes, 11 methods)
**TestComponentIntegration:**
- test_retrieval_component_base_import()
- test_retrieval_result_dataclass()
- test_pipeline_factory_import()
- test_filters_import()
- test_score_filter_functionality()
- test_limit_filter_functionality()
- test_database_controller_import()
- test_embedding_factory_import()
- test_agent_nodes_import()

**TestConfigurationValidation:**
- test_config_loader_imports()
- test_retrieval_config_structure()

### 5. test_minimal_pipeline.py (2 classes, 9 methods)
**TestMinimalPipeline:**
- test_config_loading()
- test_agent_schema_import()
- test_google_embedding_config()
- test_ci_google_config_loads()
- test_agent_retriever_config_load()
- test_pipeline_factory_google_config()

**TestConfigValidation:**
- test_config_switching()
- test_yaml_files_valid()
- test_google_embeddings_config_complete()

### 6. test_qdrant_connectivity.py (1 class, 4 methods)
**TestQdrantConnectivity:**
- test_qdrant_health_endpoint()
- test_qdrant_collections_endpoint()
- test_qdrant_collection_creation_deletion()
- test_qdrant_client_import()

### 7. test_end_to_end.py (1 class, 4 methods)
**TestEndToEndPipeline:**
- test_full_retrieval_pipeline()
- test_retrieval_ranking()
- test_config_switching_with_data()
- test_pipeline_error_handling_with_real_setup()

---

## 📦 Test Dependencies

```bash
# Install test dependencies
pip install -r tests/requirements-minimal.txt

# Required: pytest, pytest-cov, pytest-asyncio, pyyaml
```

---

## ✅ Test Coverage

```bash
# Generate HTML coverage report
python -m pytest tests/ --cov=. --cov-report=html

# View in terminal
python -m pytest tests/ --cov=. --cov-report=term

# Open coverage report
open htmlcov/index.html
```

---

## 📝 Notes

- All test files and functions listed above are **verified against the actual codebase**
- Tests use real configuration files from pipelines/configs/
- Some tests require Qdrant to be running (docker-compose up qdrant)
- Minimal tests can run without external dependencies
- No mock providers or test-specific YAML files exist

---

**Related Documentation:**
- [Components](../components/README.md)
- [Database](../database/README.md)
- [Pipelines](../pipelines/README.md)
