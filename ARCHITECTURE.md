# ReRag Architecture Patterns

This document describes the design patterns and architectural principles used in the ReRag (Retrieval-Enhanced Retrieval Augmented Generation) system.

## Table of Contents
1. [System Overview](#system-overview)
2. [Core Design Patterns](#core-design-patterns)
3. [Component Architecture](#component-architecture)
4. [Pattern Interactions](#pattern-interactions)
5. [Extension Points](#extension-points)

---

## System Overview

ReRag is a modular RAG system built with flexibility and extensibility in mind. The architecture separates concerns into distinct layers:

- **Retrieval Layer**: Handles document retrieval using dense, sparse, and hybrid strategies
- **Processing Layer**: Manages reranking, filtering, and post-processing
- **LLM Layer**: Provides multi-provider LLM integration
- **Agent Layer**: Orchestrates RAG workflows using LangGraph
- **Interface Layer**: Exposes functionality via CLI and Streamlit GUI

```
┌─────────────────────────────────────────────────────────┐
│                    Interface Layer                      │
│              (Streamlit GUI / CLI / API)                │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│                     Agent Layer                         │
│           (LangGraph State Machine)                     │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│                  Processing Pipeline                    │
│    (Chain of Responsibility + Strategy Pattern)         │
│  Retriever → Reranker → Filter → PostProcessor          │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│                  Infrastructure Layer                   │
│         (Qdrant DB / Embeddings / LLM Providers)        │
└─────────────────────────────────────────────────────────┘
```

---

## Core Design Patterns

### 1. Chain of Responsibility Pattern

**Location**: `components/retrieval_pipeline.py`

**Purpose**: Process documents through a sequential pipeline where each component can transform the results before passing to the next.

**Implementation**:
```python
class RetrievalPipeline:
    def run(self, query: str, top_k: int = 10) -> List[Document]:
        results = []
        
        for component in self.components:
            if isinstance(component, BaseRetriever):
                results = component.retrieve(query, top_k)
            else:
                results = component.process(results, query)
        
        return results
```

**Benefits**:
- Components can be added/removed dynamically
- Each component focuses on a single responsibility
- Easy to test individual components in isolation

**Example Configuration**:
```yaml
pipeline:
  - type: hybrid_retriever
  - type: cross_encoder_reranker
  - type: score_threshold_filter
  - type: metadata_enhancer
```

---

### 2. Strategy Pattern

**Location**: `retrievers/`, `components/rerankers.py`, `components/filters.py`

**Purpose**: Define a family of interchangeable algorithms (retrieval strategies, reranking strategies) that can be selected at runtime.

**Implementation**:
```python
# Abstract Strategy
class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[Document]:
        pass

# Concrete Strategies
class DenseRetriever(BaseRetriever):
    def retrieve(self, query: str, top_k: int) -> List[Document]:
        # Dense embedding-based retrieval
        pass

class SparseRetriever(BaseRetriever):
    def retrieve(self, query: str, top_k: int) -> List[Document]:
        # Sparse keyword-based retrieval (SPLADE)
        pass

class HybridRetriever(BaseRetriever):
    def retrieve(self, query: str, top_k: int) -> List[Document]:
        # Fusion of dense + sparse
        pass
```

**Benefits**:
- Runtime selection of retrieval strategy
- Easy to add new retrieval methods
- Strategies can be benchmarked against each other

---

### 3. Template Method Pattern

**Location**: `components/retrieval_pipeline.py`, `retrievers/base_retriever.py`

**Purpose**: Define the skeleton of an algorithm in a base class, letting subclasses override specific steps.

**Implementation**:
```python
class RetrievalComponent(ABC):
    """Template for all pipeline components"""
    
    @abstractmethod
    def component_name(self) -> str:
        """Subclasses must provide a name"""
        pass
    
    @abstractmethod
    def process(self, documents: List[Document], query: str) -> List[Document]:
        """Subclasses implement specific processing logic"""
        pass
    
    def validate_input(self, documents: List[Document]) -> bool:
        """Common validation logic"""
        return documents is not None and len(documents) > 0
```

**Benefits**:
- Common behavior is reused across components
- Subclasses customize only what they need
- Enforces consistent interface

---

### 4. Factory Pattern

**Location**: `config/llm_factory.py`, `embedding/factory.py`, `components/retrieval_pipeline.py`

**Purpose**: Encapsulate object creation logic and provide a unified interface for creating related objects.

**Implementation**:
```python
class LLMFactory:
    @staticmethod
    def create_llm(provider: str, model: str, **kwargs):
        """Factory method for LLM creation"""
        
        if provider == "openai":
            return _create_openai_llm(model, **kwargs)
        elif provider == "anthropic":
            return _create_anthropic_llm(model, **kwargs)
        elif provider == "gemini":
            return _create_gemini_llm(model, **kwargs)
        elif provider == "ollama":
            return _create_ollama_llm(model, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    @staticmethod
    def get_available_providers() -> List[str]:
        return ["openai", "anthropic", "gemini", "ollama"]
```

**Benefits**:
- Centralizes creation logic
- Easy to add new providers
- Hides implementation details from clients

**Example Usage**:
```python
# Create LLM without knowing implementation details
llm = LLMFactory.create_llm(
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    temperature=0.0
)
```

---

### 5. Builder Pattern

**Location**: `components/retrieval_pipeline.py` (`RetrievalPipelineFactory`)

**Purpose**: Construct complex objects step-by-step from configuration.

**Implementation**:
```python
class RetrievalPipelineFactory:
    @staticmethod
    def create_from_config(config_path: str) -> RetrievalPipeline:
        """Build pipeline from YAML configuration"""
        
        config = load_yaml_config(config_path)
        components = []
        
        for component_config in config['pipeline']:
            component = RetrievalPipelineFactory._build_component(
                component_config
            )
            components.append(component)
        
        return RetrievalPipeline(components=components)
    
    @staticmethod
    def _build_component(config: Dict) -> RetrievalComponent:
        """Build individual component from config"""
        component_type = config['type']
        
        if component_type == 'hybrid_retriever':
            return HybridRetriever(**config.get('params', {}))
        elif component_type == 'cross_encoder_reranker':
            return CrossEncoderReranker(**config.get('params', {}))
        # ... more component types
```

**Benefits**:
- Declarative pipeline configuration
- Complex object construction is separated from representation
- Same construction process can create different representations

---

### 6. Adapter Pattern

**Location**: `benchmarks/benchmarks_adapters.py`, `components/retrieval_pipeline.py`

**Purpose**: Convert interfaces of different systems to work together.

**Implementation**:
```python
class SOSumBenchmarkAdapter:
    """Adapts SOSum dataset to benchmark interface"""
    
    def adapt_query(self, row: Dict) -> str:
        """Convert dataset row to query format"""
        title = row['question_title']
        body = row.get('question_body', '')
        return f"{title} {body}".strip()
    
    def adapt_ground_truth(self, row: Dict) -> List[str]:
        """Convert dataset answers to expected format"""
        return row['accepted_answer_ids']

# LangChain Adapter
class RetrievalPipeline:
    def to_langchain_retriever(self) -> Retriever:
        """Adapt pipeline to LangChain interface"""
        return LangChainRetrieverAdapter(self)
```

**Benefits**:
- Integrates external systems (LangChain, datasets)
- Maintains clean internal interfaces
- Reusable adapters for different data sources

---

### 7. Dependency Injection

**Location**: Throughout the codebase, especially `retrievers/hybrid_retriever.py`

**Purpose**: Inject dependencies rather than creating them internally, improving testability and flexibility.

**Implementation**:
```python
class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        dense_config: Dict,
        sparse_config: Dict,
        fusion_alpha: float = 0.5,
        qdrant_db: Optional[QdrantVectorDB] = None  # Injected dependency
    ):
        self.fusion_alpha = fusion_alpha
        self._initialize_components(dense_config, sparse_config, qdrant_db)
    
    def _initialize_components(self, dense_config, sparse_config, qdrant_db):
        # Inject shared Qdrant instance into sub-retrievers
        shared_qdrant_db = qdrant_db or QdrantVectorDB(...)
        
        self.dense_retriever = DenseRetriever(
            config=dense_config,
            qdrant_db=shared_qdrant_db  # Inject
        )
        
        self.sparse_retriever = SparseRetriever(
            config=sparse_config,
            qdrant_db=shared_qdrant_db  # Inject
        )
```

**Benefits**:
- Reduces resource duplication (1 Qdrant connection instead of 4)
- Easy to mock dependencies for testing
- Explicit dependencies in constructor

---

### 8. Singleton Pattern (Implicit)

**Location**: `components/rerankers.py`, `streamlit_app.py`

**Purpose**: Ensure expensive resources (models, graphs) are loaded once and reused.

**Implementation**:
```python
# Module-level cache (implicit singleton)
_MODEL_CACHE = {}

class CrossEncoderReranker:
    def _load_model(self):
        """Load model from cache or initialize"""
        if self.model_name not in _MODEL_CACHE:
            _MODEL_CACHE[self.model_name] = CrossEncoder(
                self.model_name,
                device=self.device
            )
        return _MODEL_CACHE[self.model_name]

# Streamlit cached resource (singleton per session)
@st.cache_resource
def load_graph(agent_type: str):
    """Load agent graph once per session"""
    if agent_type == "refined":
        return graph_refined
    else:
        return graph_self_rag
```

**Benefits**:
- Avoids reloading heavy models (cross-encoders, embeddings)
- Reduces memory footprint
- Improves response time

---

### 9. Decorator Pattern

**Location**: `components/retrieval_pipeline.py`, result enhancement

**Purpose**: Add functionality to components without modifying their structure.

**Implementation**:
```python
class MetadataEnhancer(RetrievalComponent):
    """Decorates documents with additional metadata"""
    
    def process(self, documents: List[Document], query: str) -> List[Document]:
        enhanced_docs = []
        
        for doc in documents:
            # Add metadata without changing document structure
            doc.metadata['processed_at'] = datetime.now()
            doc.metadata['query'] = query
            doc.metadata['source_pipeline'] = 'rerag'
            enhanced_docs.append(doc)
        
        return enhanced_docs
```

**Benefits**:
- Extends behavior dynamically
- Preserves component interface
- Multiple decorators can be chained

---

### 10. Composite Pattern

**Location**: `retrievers/hybrid_retriever.py`

**Purpose**: Compose objects into tree structures to represent part-whole hierarchies.

**Implementation**:
```python
class HybridRetriever(BaseRetriever):
    """Composite retriever containing dense + sparse retrievers"""
    
    def retrieve(self, query: str, top_k: int) -> List[Document]:
        # Delegate to component retrievers
        dense_results = self.dense_retriever.retrieve(query, top_k)
        sparse_results = self.sparse_retriever.retrieve(query, top_k)
        
        # Fuse results
        return self._reciprocal_rank_fusion(
            dense_results, 
            sparse_results,
            alpha=self.fusion_alpha
        )
```

**Benefits**:
- Treats composite and individual objects uniformly
- Easy to create complex retrieval strategies
- Recursive composition possible

---

### 11. Null Object Pattern

**Location**: `components/retrieval_pipeline.py`

**Purpose**: Provide default behavior when optional components are missing.

**Implementation**:
```python
class NoOpReranker(Reranker):
    """Null object that does nothing"""
    
    def component_name(self) -> str:
        return "no_reranker"
    
    def process(self, documents: List[Document], query: str) -> List[Document]:
        # Pass through unchanged
        return documents

# Usage in pipeline
def build_pipeline(config):
    reranker = config.get('reranker') or NoOpReranker()
    return RetrievalPipeline(components=[retriever, reranker, filter])
```

**Benefits**:
- Eliminates None checks
- Consistent interface even with missing components
- Simplifies client code

---

## Component Architecture

### Retrieval Pipeline Components

```
RetrievalComponent (ABC)
    ├── BaseRetriever (Strategy)
    │   ├── DenseRetriever
    │   ├── SparseRetriever
    │   └── HybridRetriever (Composite)
    │
    ├── Reranker (Strategy)
    │   ├── CrossEncoderReranker
    │   ├── SemanticReranker
    │   └── NoOpReranker (Null Object)
    │
    ├── ResultFilter (Strategy)
    │   ├── ScoreThresholdFilter
    │   ├── DiversityFilter
    │   └── NoOpFilter (Null Object)
    │
    └── PostProcessor (Decorator)
        ├── MetadataEnhancer
        └── ContextExpander
```

### LLM Provider Architecture

```
LLMFactory (Factory)
    ├── _create_openai_llm()
    ├── _create_anthropic_llm()
    ├── _create_gemini_llm()
    └── _create_ollama_llm()

Providers:
    • OpenAI: GPT-4o, GPT-4o-mini
    • Anthropic: Claude 3.5 Sonnet, Claude 3 Opus/Haiku
    • Google: Gemini 1.5 Pro/Flash, Gemini 2.0 Flash
    • Ollama: Local models (llama3, mistral, etc.)
```

### Agent Architecture

```
LangGraph State Machine
    ├── Refined RAG Mode
    │   └── query → retrieve → grade → generate → final_answer
    │
    └── Self-RAG Mode
        └── query → retrieve → relevance_check → 
            ├─ if relevant → generate → support_check → final_answer
            └─ if not → web_search → generate → final_answer
```

---

## Pattern Interactions

### How Patterns Work Together

1. **Factory + Strategy**: Factory creates appropriate strategy instance
   ```python
   retriever = RetrieverFactory.create("hybrid")  # Factory creates Strategy
   ```

2. **Chain of Responsibility + Strategy**: Chain manages sequence, Strategy handles algorithm
   ```python
   pipeline = [retriever, reranker, filter]  # Chain
   # Each component uses Strategy pattern internally
   ```

3. **Builder + Template Method**: Builder constructs objects, Template Method ensures consistent interface
   ```python
   pipeline = PipelineFactory.create_from_config(config)  # Builder
   # Each component follows RetrievalComponent template
   ```

4. **Dependency Injection + Singleton**: Inject singleton instances to share resources
   ```python
   shared_db = QdrantVectorDB(...)  # Singleton instance
   retriever1 = DenseRetriever(qdrant_db=shared_db)  # Inject
   retriever2 = SparseRetriever(qdrant_db=shared_db)  # Inject
   ```

### Example: Full Pipeline Flow

```python
# 1. Factory creates LLM (Factory Pattern)
llm = LLMFactory.create_llm(provider="anthropic", model="claude-3-5-sonnet")

# 2. Builder creates pipeline from config (Builder Pattern)
pipeline = RetrievalPipelineFactory.create_from_config("hybrid_optimal.yml")

# 3. Pipeline components use Strategy Pattern
#    - HybridRetriever (Composite + Strategy)
#    - CrossEncoderReranker (Strategy + Singleton cache)
#    - ScoreThresholdFilter (Strategy)

# 4. Chain of Responsibility executes pipeline
results = pipeline.run(query="How to optimize Python code?")
#   → HybridRetriever.retrieve()  # Composite delegates to Dense + Sparse
#   → CrossEncoderReranker.process()  # Uses cached model (Singleton)
#   → ScoreThresholdFilter.process()  # Filters low-score docs

# 5. LangGraph agent orchestrates RAG workflow (State Machine)
state = {"question": "How to optimize Python code?"}
final_state = graph.invoke(state)
answer = final_state["generation"]
```

---

## Extension Points

### Adding a New Retriever

1. Extend `BaseRetriever` (Strategy Pattern)
2. Implement `retrieve()` method (Template Method)
3. Register in `RetrieverFactory` if needed
4. Add configuration schema

```python
class CustomRetriever(BaseRetriever):
    def retrieve(self, query: str, top_k: int) -> List[Document]:
        # Your custom retrieval logic
        pass
```

### Adding a New LLM Provider

1. Add provider function in `llm_factory.py` (Factory Pattern)
2. Update `create_llm()` dispatcher
3. Add to `get_available_providers()`
4. Update `config.yml` examples

```python
def _create_custom_llm(model: str, **kwargs):
    api_key = os.getenv("CUSTOM_API_KEY")
    return CustomLLM(model=model, api_key=api_key, **kwargs)
```

### Adding a New Pipeline Component

1. Extend `RetrievalComponent` (Template Method)
2. Implement `component_name()` and `process()` methods
3. Add to pipeline configuration

```python
class CustomProcessor(RetrievalComponent):
    def component_name(self) -> str:
        return "custom_processor"
    
    def process(self, documents: List[Document], query: str) -> List[Document]:
        # Your processing logic
        return documents
```

### Adding a New Agent Mode

1. Create new graph in `agent/` directory
2. Define state schema and nodes
3. Register in Streamlit GUI
4. Add configuration file

---

## Design Principles

### SOLID Principles

- **Single Responsibility**: Each component has one reason to change
  - `DenseRetriever`: only handles dense retrieval
  - `CrossEncoderReranker`: only handles reranking
  
- **Open/Closed**: Open for extension, closed for modification
  - New retrievers can be added without modifying existing code
  - Pipeline accepts any `RetrievalComponent`

- **Liskov Substitution**: Subclasses can replace base classes
  - All retrievers implement `BaseRetriever` interface
  - Any reranker can be swapped for another

- **Interface Segregation**: Clients depend only on interfaces they use
  - `RetrievalComponent` has minimal interface
  - Specialized interfaces for retrievers vs. rerankers

- **Dependency Inversion**: Depend on abstractions, not concretions
  - Pipeline depends on `RetrievalComponent` ABC
  - Factory returns interface types, not concrete classes

### DRY (Don't Repeat Yourself)

- Model caching eliminates redundant loading
- Shared Qdrant instance reduces connection overhead
- Configuration-driven pipelines avoid code duplication

### Separation of Concerns

- **Retrieval**: Focused on getting documents
- **Reranking**: Focused on scoring/ordering
- **Filtering**: Focused on selection criteria
- **LLM**: Focused on generation
- **Agent**: Focused on orchestration

---

## Performance Optimizations

### Resource Sharing (Dependency Injection)

**Before**: 4 Qdrant connections per hybrid query
```python
class HybridRetriever:
    def __init__(self):
        self.dense = DenseRetriever()  # Creates Qdrant connection
        self.sparse = SparseRetriever()  # Creates Qdrant connection
```

**After**: 1 shared connection
```python
class HybridRetriever:
    def __init__(self, qdrant_db: QdrantVectorDB):
        self.dense = DenseRetriever(qdrant_db=qdrant_db)  # Shares connection
        self.sparse = SparseRetriever(qdrant_db=qdrant_db)  # Shares connection
```

### Model Caching (Singleton Pattern)

**Before**: Reload model on every rerank
```python
class CrossEncoderReranker:
    def predict(self, query, docs):
        model = CrossEncoder(self.model_name)  # Expensive!
        return model.predict(...)
```

**After**: Load once, reuse forever
```python
_MODEL_CACHE = {}

class CrossEncoderReranker:
    def _load_model(self):
        if self.model_name not in _MODEL_CACHE:
            _MODEL_CACHE[self.model_name] = CrossEncoder(self.model_name)
        return _MODEL_CACHE[self.model_name]
```

---

## Configuration Examples

### Minimal Pipeline (Fast & Light)
```yaml
pipeline:
  - type: dense_retriever
    params:
      top_k: 20
  - type: score_threshold_filter
    params:
      threshold: 0.5
```

### Balanced Pipeline
```yaml
pipeline:
  - type: hybrid_retriever
    params:
      top_k: 30
      fusion_alpha: 0.5
  - type: cross_encoder_reranker
    params:
      model_name: ms-marco-MiniLM-L-6-v2
      top_k: 10
```

### Maximum Quality Pipeline
```yaml
pipeline:
  - type: hybrid_retriever
    params:
      top_k: 100
      fusion_alpha: 0.7
  - type: cross_encoder_reranker
    params:
      model_name: ms-marco-MiniLM-L-12-v2
      top_k: 50
  - type: semantic_reranker
    params:
      top_k: 20
  - type: diversity_filter
  - type: metadata_enhancer
```

---

## Conclusion

The ReRag system leverages multiple design patterns working in harmony to achieve:

**Modularity**: Components can be developed and tested independently  
**Extensibility**: New retrievers, rerankers, LLMs added without core changes  
**Flexibility**: Runtime configuration of pipelines and strategies  
**Performance**: Resource sharing and caching minimize overhead  
**Maintainability**: Clear separation of concerns and consistent interfaces  
**Testability**: Dependency injection enables comprehensive testing  

The architecture supports rapid experimentation with different RAG configurations while maintaining production-ready code quality.

---

## References

- **Chain of Responsibility**: `components/retrieval_pipeline.py`
- **Strategy Pattern**: `retrievers/`, `components/rerankers.py`
- **Factory Pattern**: `config/llm_factory.py`, `embedding/factory.py`
- **Builder Pattern**: `components/retrieval_pipeline.py` (RetrievalPipelineFactory)
- **Adapter Pattern**: `benchmarks/benchmarks_adapters.py`
- **Template Method**: `retrievers/base_retriever.py`, `components/retrieval_pipeline.py`
- **Dependency Injection**: `retrievers/hybrid_retriever.py`
- **Singleton Pattern**: `components/rerankers.py` (_MODEL_CACHE)
- **Decorator Pattern**: Result enhancement components
- **Composite Pattern**: `retrievers/hybrid_retriever.py`
- **Null Object Pattern**: NoOp components

For implementation details, see the respective source files.
