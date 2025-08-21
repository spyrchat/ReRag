# 🚀 System Extension and Usage Guide

## 📖 **Complete Guide to Extending and Using the RAG Retrieval System**

This guide provides step-by-step instructions for extending the modular retrieval system and using it effectively in your projects.

---

## 🎯 **Quick Start: Using the System**

### 1. **Basic Usage with Agent**

```python
# main.py - Your agent automatically uses configured pipeline
from agent.graph import graph

state = {"question": "How to handle Python exceptions?"}
result = graph.invoke(state)
print(result["answer"])
```

### 2. **Switch Retrieval Configurations**

```bash
# List available configurations
python bin/switch_agent_config.py --list

# Switch to advanced reranked pipeline  
python bin/switch_agent_config.py advanced_reranked

# Test the new configuration
python test_agent_retriever_node.py
```

### 3. **Direct Pipeline Usage (Without Agent)**

```python
from bin.agent_retriever import ConfigurableRetrieverAgent

# Load specific config
retriever = ConfigurableRetrieverAgent("pipelines/configs/retrieval/hybrid_multistage.yml")

# Search
results = retriever.search("machine learning algorithms", top_k=5)
for doc in results:
    print(f"Score: {doc.metadata.get('score', 'N/A')}")
    print(f"Content: {doc.page_content[:200]}...")
```

---

## 🔧 **Extension Guide: Adding New Components**

### **Adding a New Reranker**

#### 1. **Create Your Reranker Class**

```python
# components/my_custom_reranker.py
from typing import List
from langchain_core.documents import Document
from .rerankers import BaseReranker

class MyCustomReranker(BaseReranker):
    """Custom reranker using your preferred model/method."""
    
    def __init__(self, model_name: str = "my-model", boost_factor: float = 1.2):
        self.model_name = model_name
        self.boost_factor = boost_factor
        self.model = self._load_model()
    
    def _load_model(self):
        # Load your custom model here
        # return load_my_model(self.model_name)
        pass
    
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents using your custom logic."""
        scored_docs = []
        
        for doc in documents:
            # Your custom scoring logic
            relevance_score = self._calculate_relevance(query, doc.page_content)
            
            # Apply boost factor
            final_score = relevance_score * self.boost_factor
            
            # Update document metadata
            doc.metadata["score"] = final_score
            doc.metadata["reranker"] = f"custom_{self.model_name}"
            scored_docs.append(doc)
        
        # Sort by score (highest first)
        return sorted(scored_docs, key=lambda x: x.metadata["score"], reverse=True)
    
    def _calculate_relevance(self, query: str, content: str) -> float:
        """Implement your custom relevance calculation."""
        # Example: simple keyword matching (replace with your logic)
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        overlap = len(query_words.intersection(content_words))
        return overlap / len(query_words) if query_words else 0.0
```

#### 2. **Register Your Reranker**

```python
# components/advanced_rerankers.py
from .my_custom_reranker import MyCustomReranker

# Add to the factory function
def create_reranker(config: dict) -> BaseReranker:
    """Factory function to create rerankers."""
    reranker_type = config.get("model_type", "cross_encoder")
    
    # ... existing types ...
    
    elif reranker_type == "my_custom":
        return MyCustomReranker(
            model_name=config.get("model_name", "my-model"),
            boost_factor=config.get("boost_factor", 1.2)
        )
    
    # ... rest of function ...
```

#### 3. **Create a Configuration File**

```yaml
# pipelines/configs/retrieval/custom_reranked.yml
retrieval_pipeline:
  retriever:
    type: hybrid
    top_k: 20
    
  stages:
    - type: reranker
      config:
        model_type: my_custom
        model_name: "my-awesome-model"
        boost_factor: 1.5
        
    - type: answer_enhancer
      config:
        boost_factor: 2.0
```

#### 4. **Test Your New Component**

```bash
# Switch to your custom config
python bin/switch_agent_config.py custom_reranked

# Test it
python test_agent_retriever_node.py
```

---

### **Adding a New Filter**

#### 1. **Create Your Filter Class**

```python
# components/my_custom_filter.py
from typing import List
from langchain_core.documents import Document
from .filters import BaseFilter

class ContentLengthFilter(BaseFilter):
    """Filter documents by content length."""
    
    def __init__(self, min_length: int = 100, max_length: int = 5000):
        self.min_length = min_length
        self.max_length = max_length
    
    def filter(self, documents: List[Document]) -> List[Document]:
        """Filter documents by content length."""
        filtered_docs = []
        
        for doc in documents:
            content_length = len(doc.page_content)
            
            if self.min_length <= content_length <= self.max_length:
                # Mark as passed filter
                doc.metadata["passed_length_filter"] = True
                doc.metadata["content_length"] = content_length
                filtered_docs.append(doc)
        
        return filtered_docs

class TopicFilter(BaseFilter):
    """Filter documents by topic relevance."""
    
    def __init__(self, required_topics: List[str], topic_threshold: float = 0.3):
        self.required_topics = [topic.lower() for topic in required_topics]
        self.topic_threshold = topic_threshold
    
    def filter(self, documents: List[Document]) -> List[Document]:
        """Filter documents that contain required topics."""
        filtered_docs = []
        
        for doc in documents:
            content_lower = doc.page_content.lower()
            
            # Calculate topic relevance
            topic_matches = sum(1 for topic in self.required_topics if topic in content_lower)
            topic_score = topic_matches / len(self.required_topics)
            
            if topic_score >= self.topic_threshold:
                doc.metadata["topic_score"] = topic_score
                doc.metadata["matched_topics"] = [t for t in self.required_topics if t in content_lower]
                filtered_docs.append(doc)
        
        return filtered_docs
```

#### 2. **Register Your Filter**

```python
# components/filters.py
from .my_custom_filter import ContentLengthFilter, TopicFilter

def create_filter(config: dict) -> BaseFilter:
    """Factory function to create filters."""
    filter_type = config.get("type")
    
    # ... existing filters ...
    
    elif filter_type == "content_length":
        return ContentLengthFilter(
            min_length=config.get("min_length", 100),
            max_length=config.get("max_length", 5000)
        )
    
    elif filter_type == "topic":
        return TopicFilter(
            required_topics=config.get("topics", []),
            topic_threshold=config.get("threshold", 0.3)
        )
```

#### 3. **Use in Configuration**

```yaml
# pipelines/configs/retrieval/filtered_pipeline.yml
retrieval_pipeline:
  retriever:
    type: dense
    top_k: 20
    
  stages:
    - type: filter
      config:
        type: content_length
        min_length: 200
        max_length: 3000
        
    - type: filter
      config:
        type: topic
        topics: ["python", "machine learning", "api"]
        threshold: 0.5
        
    - type: reranker
      config:
        model_type: cross_encoder
```

---

### **Adding a New Retriever Type**

#### 1. **Create Your Retriever**

```python
# retrievers/semantic_retriever.py
from typing import List
from langchain_core.documents import Document
from .base import BaseRetriever

class SemanticRetriever(BaseRetriever):
    """Retriever that uses semantic similarity with custom embeddings."""
    
    def __init__(self, embedding_model, vector_store, similarity_threshold: float = 0.7):
        super().__init__()
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold
    
    def get_relevant_documents(self, query: str, top_k: int = 10) -> List[Document]:
        """Retrieve semantically similar documents."""
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Search vector store
        results = self.vector_store.similarity_search_with_score(
            query, 
            k=top_k * 2,  # Get more initially for filtering
            score_threshold=self.similarity_threshold
        )
        
        # Process results
        documents = []
        for doc, score in results[:top_k]:
            doc.metadata["retrieval_score"] = score
            doc.metadata["retriever_type"] = "semantic"
            doc.metadata["similarity_threshold"] = self.similarity_threshold
            documents.append(doc)
        
        return documents
```

#### 2. **Register in Pipeline Configuration**

```python
# components/retrieval_pipeline.py
from retrievers.semantic_retriever import SemanticRetriever

def create_retriever(config: dict, embedder, vectorstore):
    """Factory function to create retrievers."""
    retriever_type = config.get("type", "dense")
    
    # ... existing types ...
    
    elif retriever_type == "semantic":
        return SemanticRetriever(
            embedding_model=embedder,
            vector_store=vectorstore,
            similarity_threshold=config.get("similarity_threshold", 0.7)
        )
```

---

## 🧪 **Testing Your Extensions**

### **1. Unit Testing**

```python
# tests/test_my_extensions.py
import pytest
from langchain_core.documents import Document
from components.my_custom_reranker import MyCustomReranker
from components.my_custom_filter import ContentLengthFilter

def test_custom_reranker():
    """Test custom reranker functionality."""
    reranker = MyCustomReranker(boost_factor=1.5)
    
    docs = [
        Document(page_content="Python exception handling tutorial", metadata={}),
        Document(page_content="Java programming guide", metadata={})
    ]
    
    query = "Python exceptions"
    reranked = reranker.rerank(query, docs)
    
    # Should prefer Python content
    assert reranked[0].page_content.startswith("Python")
    assert "score" in reranked[0].metadata

def test_content_length_filter():
    """Test content length filter."""
    filter_obj = ContentLengthFilter(min_length=50, max_length=200)
    
    docs = [
        Document(page_content="Short", metadata={}),  # Too short
        Document(page_content="This is a medium length document with enough content", metadata={}),  # Good
        Document(page_content="This is a very long document " * 100, metadata={})  # Too long
    ]
    
    filtered = filter_obj.filter(docs)
    assert len(filtered) == 1
    assert "content_length" in filtered[0].metadata
```

### **2. Integration Testing**

```python
# tests/test_custom_pipeline.py
def test_custom_pipeline_end_to_end():
    """Test complete pipeline with custom components."""
    from bin.agent_retriever import ConfigurableRetrieverAgent
    
    # Test your custom configuration
    retriever = ConfigurableRetrieverAgent("pipelines/configs/retrieval/custom_reranked.yml")
    
    results = retriever.search("Python programming", top_k=5)
    
    # Verify custom components worked
    assert len(results) > 0
    assert "score" in results[0].metadata
    assert results[0].metadata.get("reranker") == "custom_my-model"
```

### **3. Run All Tests**

```bash
# Run your specific tests
python -m pytest tests/test_my_extensions.py -v

# Run all tests
python tests/run_all_tests.py
```

---

## 📊 **Performance Optimization Tips**

### **1. Efficient Reranking**

```python
# Batch processing for better performance
class BatchedReranker(BaseReranker):
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        # Process documents in batches
        batch_size = 32
        scored_docs = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_scores = self._score_batch(query, batch)
            
            for doc, score in zip(batch, batch_scores):
                doc.metadata["score"] = score
                scored_docs.append(doc)
        
        return sorted(scored_docs, key=lambda x: x.metadata["score"], reverse=True)
```

### **2. Caching for Repeated Queries**

```python
from functools import lru_cache

class CachedReranker(BaseReranker):
    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
    
    @lru_cache(maxsize=1000)
    def _cached_score(self, query: str, content: str) -> float:
        """Cache scores for repeated query-content pairs."""
        return self._calculate_relevance(query, content)
```

---

## 🔄 **A/B Testing and Experimentation**

### **1. Configuration Comparison**

```python
# scripts/compare_configs.py
from bin.agent_retriever import ConfigurableRetrieverAgent

def compare_configurations(queries: List[str]):
    """Compare different pipeline configurations."""
    configs = [
        "basic_dense",
        "advanced_reranked", 
        "hybrid_multistage",
        "custom_reranked"
    ]
    
    results = {}
    
    for config_name in configs:
        retriever = ConfigurableRetrieverAgent(f"pipelines/configs/retrieval/{config_name}.yml")
        config_results = []
        
        for query in queries:
            docs = retriever.search(query, top_k=5)
            config_results.append({
                "query": query,
                "num_results": len(docs),
                "avg_score": sum(d.metadata.get("score", 0) for d in docs) / len(docs) if docs else 0,
                "top_score": max(d.metadata.get("score", 0) for d in docs) if docs else 0
            })
        
        results[config_name] = config_results
    
    return results

# Usage
test_queries = [
    "Python exception handling",
    "machine learning algorithms", 
    "database optimization techniques"
]

comparison = compare_configurations(test_queries)
```

### **2. Evaluation Metrics**

```python
# evaluation/metrics.py
def calculate_retrieval_metrics(results: List[Document], ground_truth: List[str]) -> dict:
    """Calculate standard retrieval metrics."""
    retrieved_ids = {doc.metadata.get("id") for doc in results}
    relevant_ids = set(ground_truth)
    
    # Precision and Recall
    precision = len(retrieved_ids & relevant_ids) / len(retrieved_ids) if retrieved_ids else 0
    recall = len(retrieved_ids & relevant_ids) / len(relevant_ids) if relevant_ids else 0
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "num_retrieved": len(retrieved_ids),
        "num_relevant": len(relevant_ids)
    }
```

---

## 🚀 **Production Deployment Tips**

### **1. Environment Configuration**

```yaml
# production_config.yml
retrieval:
  config_path: "pipelines/configs/retrieval/production_optimized.yml"

# Production optimized pipeline
retrieval_pipeline:
  retriever:
    type: hybrid
    top_k: 10  # Smaller for speed
    
  stages:
    - type: reranker
      config:
        model_type: cross_encoder
        model_name: "ms-marco-MiniLM-L-2-v2"  # Faster model
        
    - type: filter
      config:
        type: score
        min_score: 0.5  # Filter low-quality results
```

### **2. Monitoring and Logging**

```python
# monitoring/retrieval_monitor.py
import time
import logging
from typing import List
from langchain_core.documents import Document

class RetrievalMonitor:
    def __init__(self):
        self.logger = logging.getLogger("retrieval_monitor")
    
    def log_retrieval(self, query: str, results: List[Document], latency: float):
        """Log retrieval performance metrics."""
        self.logger.info(f"Query: {query[:100]}...")
        self.logger.info(f"Results: {len(results)} documents")
        self.logger.info(f"Latency: {latency:.3f}s")
        
        if results:
            scores = [doc.metadata.get("score", 0) for doc in results]
            self.logger.info(f"Score range: {min(scores):.3f} - {max(scores):.3f}")

# Use in your pipeline
monitor = RetrievalMonitor()

def monitored_search(retriever, query: str, top_k: int = 5):
    start_time = time.time()
    results = retriever.search(query, top_k)
    latency = time.time() - start_time
    
    monitor.log_retrieval(query, results, latency)
    return results
```

---

## 🎯 **Common Extension Patterns**

### **1. Multi-Modal Reranking**

```python
class MultiModalReranker(BaseReranker):
    """Rerank using both text and metadata signals."""
    
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        for doc in documents:
            text_score = self._text_relevance(query, doc.page_content)
            metadata_score = self._metadata_relevance(query, doc.metadata)
            
            # Combine scores with weights
            final_score = 0.7 * text_score + 0.3 * metadata_score
            doc.metadata["score"] = final_score
            doc.metadata["text_score"] = text_score
            doc.metadata["metadata_score"] = metadata_score
        
        return sorted(documents, key=lambda x: x.metadata["score"], reverse=True)
```

### **2. Adaptive Top-K Selection**

```python
class AdaptiveRetriever:
    """Dynamically adjust top_k based on query complexity."""
    
    def get_adaptive_top_k(self, query: str, base_k: int = 10) -> int:
        """Calculate adaptive top_k based on query characteristics."""
        query_words = len(query.split())
        
        if query_words <= 3:
            return base_k // 2  # Simple queries need fewer results
        elif query_words >= 10:
            return base_k * 2   # Complex queries need more results
        else:
            return base_k
```

### **3. Context-Aware Filtering**

```python
class ContextAwareFilter(BaseFilter):
    """Filter based on conversation context."""
    
    def __init__(self, conversation_history: List[str]):
        self.conversation_history = conversation_history
    
    def filter(self, documents: List[Document]) -> List[Document]:
        # Extract topics from conversation history
        context_topics = self._extract_topics(self.conversation_history)
        
        filtered_docs = []
        for doc in documents:
            context_relevance = self._calculate_context_relevance(doc, context_topics)
            
            if context_relevance > 0.3:
                doc.metadata["context_relevance"] = context_relevance
                filtered_docs.append(doc)
        
        return filtered_docs
```

---

## 📚 **Resources and Further Reading**

### **Documentation Files**
- `docs/AGENT_INTEGRATION.md` - How the agent integrates with pipelines
- `docs/EXTENSIBILITY.md` - Quick extensibility overview  
- `docs/MLOPS_PIPELINE_ARCHITECTURE.md` - System architecture
- `pipelines/configs/retrieval/` - Example configurations

### **Key Code Files**
- `components/retrieval_pipeline.py` - Main pipeline implementation
- `components/rerankers.py` - Reranker implementations
- `components/filters.py` - Filter implementations
- `bin/agent_retriever.py` - Configurable retriever agent
- `agent/nodes/retriever.py` - Agent integration

### **Testing and Examples**
- `test_agent_retriever_node.py` - Test agent integration
- `tests/retrieval/` - Comprehensive test suite
- `examples/` - Usage examples
- `bin/switch_agent_config.py` - Configuration management

---

## 🎉 **You're Ready to Extend!**

With this guide, you can:
- ✅ Add custom rerankers, filters, and retrievers
- ✅ Create new pipeline configurations  
- ✅ Test and evaluate your extensions
- ✅ Deploy optimized pipelines to production
- ✅ Monitor and improve performance

**Happy extending!** 🚀
