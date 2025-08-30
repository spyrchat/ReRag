# Pipeline Tests

This directory contains comprehensive tests for the RAG pipeline with three levels of testing:

## 🎯 **Test Levels**

### **Level 1: Minimal Tests** (No External Dependencies)
- ✅ Configuration loading and validation
- ✅ Component imports and integration
- ✅ YAML syntax validation
- ✅ Google embeddings configuration checks
- ✅ Agent schema validation

**Usage**: Perfect for CI/CD, development, and basic validation
```bash
python -m pytest tests/pipeline/test_minimal_pipeline.py tests/pipeline/test_components.py -v
```

### **Level 2: Integration Tests** (Requires Qdrant)
- ✅ Qdrant connectivity and health checks
- ✅ Collection creation/deletion
- ✅ Basic database operations
- ✅ Client library functionality

**Usage**: For testing with database services
```bash
python -m pytest tests/pipeline/test_qdrant_connectivity.py -v
```

### **Level 3: End-to-End Tests** (Requires API Key + Qdrant + Data)
- ✅ Complete pipeline execution with real embeddings
- ✅ Automatic test collection setup with sample documents
- ✅ Real retrieval queries and ranking validation
- ✅ Configuration switching with actual data
- ✅ Error handling with live services

**Usage**: For full pipeline validation
```bash
GOOGLE_API_KEY=your_key python -m pytest tests/pipeline/test_end_to_end.py -v -m "requires_api"
```

## 📁 **Test Files**

| File | Purpose | Dependencies | Description |
|------|---------|--------------|-------------|
| `test_minimal_pipeline.py` | Core validation | None | Config loading, imports, structure validation |
| `test_components.py` | Component integration | None | Filter functionality, data structures, imports |
| `test_qdrant_connectivity.py` | Database connectivity | Qdrant | Health checks, collection operations |
| `test_end_to_end.py` | Complete pipeline | API Key + Qdrant | Real retrieval with sample data |
| `run_tests.py` | Test runner | Auto-detect | Comprehensive test suite with reporting |

## 🚀 **Quick Start**

### **Local Development**
```bash
# 1. Install test dependencies
pip install -r tests/requirements-test.txt

# 2. Run minimal tests (always work)
python tests/pipeline/run_tests.py

# 3. With Qdrant (optional)
docker run -p 6333:6333 qdrant/qdrant
python tests/pipeline/run_tests.py

# 4. With API key (full pipeline)
export GOOGLE_API_KEY=your_api_key
python tests/pipeline/run_tests.py
```

### **GitHub Actions CI/CD**
The pipeline automatically runs in GitHub Actions with three jobs:

1. **Minimal Tests**: Always run, no external dependencies
2. **Integration Tests**: Run with Qdrant service container
3. **End-to-End Tests**: Run with your `GOOGLE_API_KEY` secret

**Setup**:
1. Add `GOOGLE_API_KEY` as a GitHub repository secret
2. Push your code - tests run automatically
3. Check Actions tab for results

## 🎯 **Test Coverage**

### **✅ What We Test**

#### **Configuration (9 tests)**
- YAML file validity across the project
- Google embeddings configuration completeness
- Required fields presence and correctness
- Config structure and nesting
- Agent schema compatibility

#### **Components (11 tests)**
- Pipeline component imports and initialization
- Filter functionality (ScoreFilter, ResultLimiter)
- Data structure integrity (RetrievalResult, Document)
- Database controller instantiation
- Factory method availability

#### **Connectivity (4 tests)**
- Qdrant service health and endpoints
- Collection lifecycle (create/list/delete)
- Client library integration
- Connection error handling

#### **End-to-End (4 tests)**
- **Complete Pipeline Flow**: Query → Embedding → Search → Results
- **Test Data Setup**: Automatic creation of test collection with 5 sample documents:
  - Python exception handling guide
  - Binary search algorithm explanation
  - Machine learning introduction
  - REST API design principles
  - Docker container fundamentals
- **Real Retrieval Validation**:
  - Query: "How to handle errors in Python?" → Returns Python exceptions doc
  - Query: "What is binary search algorithm?" → Returns algorithm doc
  - Query: "Machine learning basics" → Returns ML introduction
- **Quality Checks**:
  - Results ranked by relevance score
  - Minimum score thresholds met
  - Expected keywords found in top results
  - Result structure completeness

### **❌ Deliberately Avoided**
- Local model downloads (sentence transformers, etc.)
- Heavy GPU/compute requirements
- Large file dependencies
- Network-dependent operations in basic tests
- Hardcoded API keys or secrets

## 🔧 **Configuration**

### **Environment Variables**
```bash
# Required for end-to-end tests
GOOGLE_API_KEY=your_google_api_key_here

# Optional (auto-detected)
QDRANT_HOST=localhost      # Default: localhost
QDRANT_PORT=6333           # Default: 6333
```

### **GitHub Secrets**
Set in your repository: Settings → Secrets and variables → Actions

| Secret Name | Description | Required For |
|-------------|-------------|--------------|
| `GOOGLE_API_KEY` | Google Gemini API key | End-to-end tests |

### **Test Markers**
Use pytest markers to run specific test types:

```bash
# Run only integration tests
python -m pytest -m "integration" -v

# Run only API-requiring tests  
python -m pytest -m "requires_api" -v

# Run everything except API tests
python -m pytest -m "not requires_api" -v

# Run end-to-end tests only
python -m pytest tests/pipeline/test_end_to_end.py -v
```

## 💡 **Best Practices**

### **Development Workflow**
1. **Start with minimal tests** - Always pass before moving to integration
2. **Test locally first** - Use `python tests/test_local_setup.py`
3. **Use progressive testing** - Level 1 → Level 2 → Level 3
4. **Check CI early** - Don't wait for complex features to test CI setup

### **CI/CD Integration**
1. **Minimal tests in all PRs** - Fast feedback on basic issues
2. **Integration tests on main/develop** - Catch service integration issues  
3. **End-to-end tests on releases** - Full validation before deployment
4. **Security validation** - No hardcoded secrets, proper env var usage

This test suite ensures your RAG pipeline works correctly across all environments! 🎯
