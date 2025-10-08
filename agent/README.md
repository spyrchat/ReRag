# Agent Module

LangGraph-powered agent workflows for intelligent query processing and response generation in RAG systems.

## 📋 Overview

The agent module implements sophisticated AI workflows using LangGraph, providing:

- **Intelligent Query Processing**: Multi-step query interpretation and planning
- **Dynamic Retrieval**: Context-aware document retrieval with strategy selection
- **Response Generation**: High-quality answers with source attribution
- **Workflow Orchestration**: Configurable agent graphs with conditional logic
- **State Management**: Persistent conversation state and context tracking

## 🏗️ Architecture

```
agent/
├── __init__.py
├── graph.py                    # Agent workflow (LangGraph)
├── schema.py                   # Data models and state
├── nodes/                      # Agent nodes
│   ├── query_interpreter.py    # Query analysis
│   ├── retriever.py            # Document retrieval
│   ├── generator.py            # Response generation
│   └── memory_updater.py       # Memory/state updates
└── README.md                   # This file
```

### Workflow Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Agent Workflow (LangGraph)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  📝 Query Input                                                     │
│      ↓                                                              │
│  🧠 Query Interpreter                                               │
│      ├─ Intent Classification                                       │
│      ├─ Query Expansion                                             │
│      └─ Strategy Selection                                          │
│      ↓                                                              │
│  🔍 Retrieval Node                                                  │
│      ├─ Vector Search (Dense/Sparse/Hybrid)                         │
│      ├─ Metadata Filtering                                          │
│      └─ Multi-hop Retrieval                                         │
│      ↓                                                              │
│  🎯 Context Filter                                                  │
│      ├─ Relevance Scoring                                           │
│      ├─ Deduplication                                               │
│      └─ Context Ranking                                             │
│      ↓                                                              │
│  🤖 Response Generator                                              │
│      ├─ Context Integration                                         │
│      ├─ Answer Generation                                           │
│      └─ Source Attribution                                          │
│      ↓                                                              │
│  ✅ Quality Checker                                                 │
│      ├─ Factual Verification                                        │
│      ├─ Completeness Check                                          │
│      └─ Final Response                                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Basic Usage

```python
from agent.graph import graph
from agent.schema import AgentState

# Use the pre-built agent workflow
initial_state = {
    "query": "What are the benefits of renewable energy?",
    "conversation_history": []
}

# Run workflow
result = graph.invoke(initial_state)
print(f"Answer: {result['response']}")
```

### Configuration-Based Setup

```python
import yaml
from agent.graph import graph

# Load configuration
with open("config/agent_config.yml") as f:
    config = yaml.safe_load(f)

# Create configured agent
agent = graph(config=config)

# Process query
result = agent.invoke({
    "query": "How do solar panels work?",
    "conversation_id": "user_123",
    "context": {"domain": "renewable_energy"}
})
```

### Interactive Mode

```python
from agent.graph import graph

agent = graph

while True:
    query = input("Ask a question (or 'quit'): ")
    if query.lower() == 'quit':
        break
    
    result = agent.invoke({"query": query})
    print(f"\nAnswer: {result['response']}\n")
    
    # Show sources
    for i, doc in enumerate(result['retrieved_docs']):
        print(f"Source {i+1}: {doc.metadata.get('source', 'Unknown')}")
```

## ⚙️ Configuration

### Agent Configuration

```yaml
# config/agent_config.yml
agent:
  llm:
    provider: "openai"
    model: "gpt-4"
    temperature: 0.1
    max_tokens: 2000
  
  retrieval:
    strategy: "hybrid"
    max_docs: 10
    min_relevance_score: 0.7
    rerank: true
  
  response:
    include_sources: true
    max_response_length: 1000
    confidence_threshold: 0.8

# Database configuration
database:
  collection: "knowledge_base"
  embedding_strategy: "hybrid"

# Embedding configuration  
embedding:
  dense_provider: "google"
  sparse_provider: "splade"
```

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `LLM_PROVIDER` | LLM provider (openai, google) | `openai` | Yes |
| `LLM_MODEL` | Model name | `gpt-4` | No |
| `OPENAI_API_KEY` | OpenAI API key | - | If using OpenAI |
| `GOOGLE_API_KEY` | Google AI API key | - | If using Google |
| `AGENT_TEMPERATURE` | LLM temperature | `0.1` | No |
| `MAX_DOCS_RETRIEVED` | Max documents per query | `10` | No |

## 🧠 Agent Nodes

### Query Interpreter

Analyzes incoming queries and plans retrieval strategy.

```python
from agent.nodes.query_interpreter import query_interpreter_node

# Features:
# - Intent classification (factual, how-to, comparison, etc.)
# - Query expansion with synonyms and related terms
# - Strategy selection (dense vs sparse vs hybrid)
# - Context extraction from conversation history
```

**Capabilities:**
- **Intent Detection**: Classifies query types (factual, procedural, comparative)
- **Query Expansion**: Adds related terms and synonyms
- **Strategy Selection**: Chooses optimal retrieval strategy
- **Context Awareness**: Incorporates conversation history

### Retrieval Node

Performs intelligent document retrieval with multiple strategies.

```python
from agent.nodes.retriever_node import retrieval_node

# Features:
# - Multi-strategy search (dense, sparse, hybrid)
# - Metadata filtering
# - Multi-hop retrieval for complex queries
# - Automatic fallback strategies
```

**Capabilities:**
- **Hybrid Search**: Combines semantic and keyword matching
- **Metadata Filtering**: Filters by source, date, category, etc.
- **Multi-hop Retrieval**: Follows references for complex queries
- **Adaptive Strategy**: Adjusts search based on initial results

### Context Filter

Ranks and filters retrieved documents for relevance.

```python
from agent.nodes.context_filter import context_filter_node

# Features:
# - Relevance scoring with multiple algorithms
# - Deduplication of similar content
# - Context window optimization
# - Quality-based ranking
```

**Capabilities:**
- **Relevance Scoring**: Multiple scoring algorithms (BM25, semantic similarity)
- **Deduplication**: Removes near-duplicate content
- **Context Optimization**: Fits important content in LLM context window
- **Quality Filtering**: Removes low-quality or irrelevant documents

### Response Generator

Generates comprehensive answers with source attribution.

```python
from agent.nodes.response_generator import response_generator_node

# Features:
# - Context-aware answer generation
# - Source attribution and citations
# - Multiple response formats
# - Confidence scoring
```

**Capabilities:**
- **Contextual Generation**: Integrates retrieved context naturally
- **Source Attribution**: Provides clear citations and references
- **Format Adaptation**: Adjusts response style based on query type
- **Confidence Estimation**: Provides confidence scores for answers

### Quality Checker

Validates response quality and completeness.

```python
from agent.nodes.quality_checker import quality_checker_node

# Features:
# - Factual consistency checking
# - Completeness validation
# - Source verification
# - Response refinement
```

**Capabilities:**
- **Fact Checking**: Verifies claims against source documents
- **Completeness Check**: Ensures all aspects of query are addressed
- **Source Validation**: Confirms proper source attribution
- **Refinement**: Suggests improvements or requests more information

## 🔧 Advanced Features

### Conversation Memory

```python
from agent.schema import AgentState

# Maintain conversation context
conversation_state = AgentState(
    query="What is solar energy?",
    conversation_id="user_123",
    conversation_history=[
        {"role": "user", "content": "Tell me about renewable energy"},
        {"role": "assistant", "content": "Renewable energy comes from..."}
    ]
)

result = agent.invoke(conversation_state)
```

### Custom Workflows

```python
from langgraph.graph import StateGraph
from agent.schema import AgentState
from agent.nodes import *

# Create custom workflow
workflow = StateGraph(AgentState)

# Add custom nodes
workflow.add_node("custom_preprocessor", custom_preprocess_node)
workflow.add_node("query_interpreter", query_interpreter_node)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("custom_filter", custom_filter_node)
workflow.add_node("response_generator", response_generator_node)

# Define custom flow
workflow.add_edge("custom_preprocessor", "query_interpreter")
workflow.add_edge("query_interpreter", "retrieval")
workflow.add_conditional_edges(
    "retrieval",
    lambda state: "custom_filter" if len(state["retrieved_docs"]) > 10 else "response_generator"
)

agent = workflow.compile()
```

### Conditional Logic

```python
def should_expand_query(state: AgentState) -> str:
    """Conditional logic for query expansion"""
    if len(state["retrieved_docs"]) < 3:
        return "expand_query"
    elif state["query_intent"] == "comparison":
        return "multi_retrieval"
    else:
        return "context_filter"

# Add conditional edges
workflow.add_conditional_edges("retrieval", should_expand_query)
```

### Streaming Responses

```python
from agent.graph import graph

agent = graph

# Stream response generation
for chunk in agent.stream({"query": "How do wind turbines work?"}):
    if "response_generator" in chunk:
        print(chunk["response_generator"]["partial_response"], end="")
```

## 📊 Monitoring & Debugging

### Execution Tracing

```python
from agent.graph import graph

agent = graph(debug=True)

# Run with detailed tracing
result = agent.invoke(
    {"query": "What is photosynthesis?"},
    config={"trace": True}
)

# View execution path
for step in result["execution_trace"]:
    print(f"Node: {step['node']}, Duration: {step['duration']:.2f}s")
```

### Performance Metrics

```python
import time
from agent.graph import graph

agent = graph

# Measure performance
start_time = time.time()
result = agent.invoke({"query": "Benefits of electric vehicles"})
total_time = time.time() - start_time

print(f"Total time: {total_time:.2f}s")
print(f"Documents retrieved: {len(result['retrieved_docs'])}")
print(f"Response length: {len(result['response'])} chars")
print(f"Confidence: {result.get('confidence', 'N/A')}")
```

### Error Handling

```python
from agent.graph import graph
from agent.schema import AgentState

agent = graph

try:
    result = agent.invoke({
        "query": "Complex technical question",
        "max_retries": 3,
        "fallback_strategy": "simplified"
    })
except Exception as e:
    print(f"Agent workflow failed: {e}")
    # Implement fallback logic
```

## 🔌 Extension Points

### Adding Custom Nodes

1. **Create Node Function**
   ```python
   from agent.schema import AgentState
   
   def my_custom_node(state: AgentState) -> AgentState:
       # Your custom logic here
       state["custom_data"] = process_custom_logic(state["query"])
       return state
   ```

2. **Add to Workflow**
   ```python
   from agent.graph import graph
   
   def create_custom_agent():
       workflow = StateGraph(AgentState)
       
       # Add standard nodes
       workflow.add_node("query_interpreter", query_interpreter_node)
       workflow.add_node("my_custom_node", my_custom_node)  # Add custom node
       workflow.add_node("retrieval", retrieval_node)
       
       # Define flow
       workflow.add_edge("query_interpreter", "my_custom_node")
       workflow.add_edge("my_custom_node", "retrieval")
       
       return workflow.compile()
   ```

### Custom LLM Providers

```python
from langchain_core.language_models import BaseLLM

class MyCustomLLM(BaseLLM):
    def _call(self, prompt: str, **kwargs) -> str:
        # Your custom LLM implementation
        return response

# Use in agent configuration
config = {
    "llm": {
        "provider": "custom",
        "instance": MyCustomLLM()
    }
}
```

### Custom Retrieval Strategies

```python
from agent.nodes.retriever_node import BaseRetriever

class MyCustomRetriever(BaseRetriever):
    def retrieve(self, query: str, **kwargs) -> List[Document]:
        # Your custom retrieval logic
        return documents

# Register custom retriever
RETRIEVER_REGISTRY["my_strategy"] = MyCustomRetriever
```

## 🧪 Testing

### Unit Tests

```bash
# Test individual nodes
pytest tests/unit/test_agent_nodes.py -v

# Test specific node
pytest tests/unit/test_agent_nodes.py::test_query_interpreter -v
```

### Integration Tests

```bash
# Test full workflow (requires vector DB)
docker-compose up -d qdrant
pytest tests/integration/test_agent_workflow.py -v
```

### End-to-End Tests

```bash
# Test with real data and APIs
export OPENAI_API_KEY=your_key
export GOOGLE_API_KEY=your_key
pytest tests/e2e/test_agent_e2e.py -v
```

### Manual Testing

```python
# Interactive testing
from agent.graph import graph

agent = graph

test_queries = [
    "What is renewable energy?",
    "Compare solar vs wind power",
    "How do you install solar panels?",
    "What are the costs of renewable energy?"
]

for query in test_queries:
    print(f"\nQuery: {query}")
    result = agent.invoke({"query": query})
    print(f"Response: {result['response'][:200]}...")
    print(f"Sources: {len(result['retrieved_docs'])}")
```

## 🚨 Troubleshooting

### Common Issues

1. **LLM API Errors**
   ```
   Error: Invalid API key
   ```
   **Solution**: Check `OPENAI_API_KEY` or `GOOGLE_API_KEY` environment variables

2. **No Retrieved Documents**
   ```
   Warning: No documents retrieved for query
   ```
   **Solution**: Check vector database connection and embedding configuration

3. **Memory Issues**
   ```
   Error: Context window exceeded
   ```
   **Solution**: Reduce `max_docs` or implement better context filtering

4. **Slow Response Times**
   ```
   Warning: Query took 30+ seconds
   ```
   **Solution**: Optimize retrieval parameters or use caching

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from agent.graph import graph

# Enable detailed logging
agent = graph(debug=True, verbose=True)
```

### Performance Optimization

```python
# Optimize for speed
config = {
    "retrieval": {
        "max_docs": 5,  # Reduce documents
        "early_stopping": True,
        "cache_enabled": True
    },
    "llm": {
        "temperature": 0.0,  # Deterministic responses
        "max_tokens": 500    # Shorter responses
    }
}

agent = graph(config=config)
```

## 📈 Best Practices

### Production Deployment

1. **Error Handling**
   ```python
   def robust_agent_call(query: str, max_retries: int = 3):
       for attempt in range(max_retries):
           try:
               return agent.invoke({"query": query})
           except Exception as e:
               if attempt == max_retries - 1:
                   raise
               time.sleep(2 ** attempt)  # Exponential backoff
   ```

2. **Response Caching**
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)
   def cached_agent_call(query: str) -> str:
       result = agent.invoke({"query": query})
       return result["response"]
   ```

3. **Monitoring**
   ```python
   from logs.utils.logger import get_logger
   
   logger = get_logger(__name__)
   
   def monitored_agent_call(query: str):
       start_time = time.time()
       try:
           result = agent.invoke({"query": query})
           duration = time.time() - start_time
           
           logger.info(f"Agent query completed", extra={
               "query_length": len(query),
               "response_length": len(result["response"]),
               "docs_retrieved": len(result["retrieved_docs"]),
               "duration": duration,
               "success": True
           })
           return result
       except Exception as e:
           duration = time.time() - start_time
           logger.error(f"Agent query failed", extra={
               "query_length": len(query),
               "duration": duration,
               "error": str(e),
               "success": False
           })
           raise
   ```

### Quality Assurance

- **Test with diverse query types** (factual, how-to, comparison)
- **Validate source attribution** accuracy
- **Monitor response coherence** and relevance
- **Track user satisfaction** metrics

---

## 🔗 Related Documentation

- **[Retrievers README](../retrievers/README.md)**: Search and retrieval
- **[Database README](../database/README.md)**: Vector storage
- **[Embedding README](../embedding/README.md)**: Embedding generation
- **[Main README](../readme.md)**: System overview

## 📞 Support

For agent-specific issues:
1. Check LLM API key configuration
2. Verify vector database connectivity
3. Review query complexity and context window limits
4. Monitor performance metrics and optimize parameters
