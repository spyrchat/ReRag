# Agent Module

LangGraph-powered agent workflows for RAG query processing with two modes: **Standard RAG** and **Self-RAG** (with iterative refinement).

## 📋 Overview

The agent module provides:

- **Two Agent Modes**: Standard (single-pass) and Self-RAG (iterative refinement)
- **Query Analysis**: Intent classification and query breakdown
- **Dynamic Routing**: Conditional retrieval based on query type
- **Response Generation**: Context-aware answer generation
- **Self-Verification**: Hallucination detection and correction (Self-RAG mode)
- **State Management**: Conversation history tracking

## 🏗️ Architecture

```
agent/
├── graph_refined.py           # Standard RAG workflow
├── graph_self_rag.py          # Self-RAG workflow (with verification)
├── schema.py                  # AgentState definition
├── nodes/
│   ├── query_analyzer.py      # Query analysis
│   ├── router.py              # Routing decisions
│   ├── retriever.py           # Document retrieval
│   ├── generator.py           # Answer generation (standard)
│   ├── self_rag_generator.py  # Answer generation (self-correcting)
│   ├── verifier.py            # Hallucination detection
│   ├── memory_updater.py      # Conversation memory
│   └── benchmark_logger.py    # Execution logging
└── README.md                  # This file
```

### Workflow Comparison

**Standard RAG** (`graph_refined.py`):
```
Query → Analyzer → Router → Retriever* → Generator → Answer
                              ↓
                         (if needed)
```

**Self-RAG** (`graph_self_rag.py`):
```
Query → Analyzer → Retriever → Generator → Verifier → Answer
                                   ↑           ↓
                                   └─ Refine ─┘ (if hallucination detected)
```

**Key Differences**:
- Standard: Single-pass, optional retrieval, faster
- Self-RAG: Always retrieves, iterative refinement, more accurate

## 🚀 Quick Start

### Using via main.py (Recommended)

```bash
# Standard RAG mode
python main.py --query "What are Python best practices?"

# Self-RAG mode (iterative refinement)
python main.py --mode self-rag --query "Explain how asyncio works"

# Interactive mode
python main.py
# or
python main.py --mode self-rag
```

### Programmatic Usage

```python
# Standard RAG
from agent.graph_refined import graph

state = {
    "question": "What is recursion?",
    "chat_history": []
}

result = graph.invoke(state)
print(f"Answer: {result['answer']}")
```

```python
# Self-RAG mode
from agent.graph_self_rag import graph

state = {
    "question": "What is recursion?",
    "chat_history": []
}

result = graph.invoke(state)
print(f"Answer: {result['answer']}")
print(f"Iterations: {result.get('iteration_count', 1)}")
```

## ⚙️ Configuration

### Main Configuration (`config.yml`)

```yaml
llm:
  provider: "openai"        # or "ollama"
  model: "gpt-4o-mini"
  temperature: 0.1
  max_tokens: 2000

generation:
  prompt_style: "strict"    # or "flexible"

self_rag:
  max_iterations: 3         # Max refinement loops

benchmark:
  enabled: false            # Set true to log execution metrics

retriever:
  strategy: "hybrid"        # dense, sparse, or hybrid
  top_k: 10
  alpha: 0.7                # For hybrid: 0=sparse, 1=dense
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Yes (if using OpenAI) |
| `GOOGLE_API_KEY` | Google AI API key | Yes (if using Google) |
| `BENCHMARK_MODE` | Enable benchmark logging | No |

## 🧠 Agent Nodes

### Query Analyzer
**File**: `nodes/query_analyzer.py`

Breaks down the query into logical steps and classifies intent.

**Features**:
- Query type classification (technical/general/clarification)
- Query breakdown and analysis
- Sets reference date for temporal queries

### Router
**File**: `nodes/router.py`

Decides whether retrieval is needed (Standard RAG only).

**Features**:
- Analyzes if database lookup is required
- Routes to retriever or directly to generator
- Optimizes for simple queries that don't need context

### Retriever
**File**: `nodes/retriever.py`

Retrieves relevant documents from vector database.

**Features**:
- Uses configured retrieval strategy (dense/sparse/hybrid)
- Applies reranking if configured
- Formats context for LLM consumption
- Logs retrieval metadata

### Generator (Standard)
**File**: `nodes/generator.py`

Generates answers using retrieved context (single-pass).

**Features**:
- Context-aware generation
- Configurable prompt styles (strict/flexible)
- Source-grounded responses

### Self-RAG Generator
**File**: `nodes/self_rag_generator.py`

Generates answers with iterative refinement loop.

**Features**:
- Initial answer generation
- Hallucination detection via verifier
- Iterative refinement (up to max_iterations)
- Tracks refinement count and reasons

### Verifier
**File**: `nodes/verifier.py`

Checks generated answers for hallucinations (Self-RAG only).

**Features**:
- Compares answer against source context
- Detects unsupported claims
- Provides feedback for refinement

### Memory Updater
**File**: `nodes/memory_updater.py`

Updates conversation history (optional).

**Features**:
- Maintains chat history
- Tracks conversation state

### Benchmark Logger
**File**: `nodes/benchmark_logger.py`

Logs execution metrics for evaluation.

**Features**:
- Logs query, answer, context, metadata
- Tracks execution time and iterations
- Outputs to `logs/benchmark/` when enabled

## 🔧 Advanced Usage

### AgentState Schema

The state passed between nodes (defined in `schema.py`):

```python
from agent.schema import AgentState

state = {
    # Input
    "question": str,              # User query
    "chat_history": List[str],    # Conversation history
    
    # Analysis
    "query_analysis": str,        # Query breakdown
    "query_type": str,            # technical/general/clarification
    
    # Routing (Standard RAG only)
    "needs_retrieval": bool,      # Router decision
    
    # Retrieval
    "context": str,               # Retrieved text
    "retrieved_documents": List,  # Document objects
    "retrieval_metadata": dict,   # Scores, method, etc.
    
    # Generation
    "answer": str,                # Final answer
    "generation_mode": str,       # context/direct/error
    
    # Self-RAG specific
    "iteration_count": int,       # Number of refinements
    "verification_results": List, # Verification history
    
    # Metadata
    "reference_date": str,        # For temporal queries
    "error": str                  # Error messages
}
```

### Custom Graph Modifications

See `graph_refined.py` or `graph_self_rag.py` to modify the workflow:

```python
# Example: Add custom node to existing graph
from langgraph.graph import StateGraph
from agent.nodes.query_analyzer import make_query_analyzer
from agent.schema import AgentState

def my_custom_node(state: AgentState) -> AgentState:
    # Your logic here
    state["custom_field"] = "custom value"
    return state

# Rebuild graph with custom node
workflow = StateGraph(AgentState)
workflow.add_node("custom", my_custom_node)
# ... add other nodes and edges
graph = workflow.compile()
```

## 📊 Monitoring & Logging

### Execution Logs

Logs are written to:
- `logs/agent.log` - Main execution log
- `logs/query_interpreter.log` - Query analysis details
- `logs/benchmark/` - Benchmark mode logs (when enabled)

### Enable Benchmark Mode

Track execution metrics by setting in `config.yml`:

```yaml
benchmark:
  enabled: true
```

Or via environment variable:
```bash
export BENCHMARK_MODE=true
python main.py --query "test query"
```

Logs include:
- Query and answer
- Retrieved context
- Execution time
- Iteration count (Self-RAG)
- Retrieval metadata

## 🔌 Customization

### Add Custom Node

1. Create node function in `agent/nodes/`:
   ```python
   # agent/nodes/my_custom_node.py
   from agent.schema import AgentState
   
   def my_custom_node(state: AgentState) -> AgentState:
       # Your logic
       state["custom_field"] = "value"
       return state
   ```

2. Modify graph in `graph_refined.py` or `graph_self_rag.py`:
   ```python
   from agent.nodes.my_custom_node import my_custom_node
   
   # Add to workflow
   workflow.add_node("custom", my_custom_node)
   workflow.add_edge("query_analyzer", "custom")
   workflow.add_edge("custom", "retriever")
   ```

### Change LLM Provider

Edit `config.yml`:
```yaml
llm:
  provider: "ollama"  # or "openai"
  model: "llama3.1"
  base_url: "http://localhost:11434"  # For Ollama
```

The `config/llm_factory.py` supports OpenAI and Ollama out of the box.

## 🧪 Testing

### Integration Tests

```bash
# Start vector database
docker-compose up -d

# Run Self-RAG integration tests
pytest tests/test_self_rag_integration.py -v

# Run all agent tests
pytest tests/ -k agent -v
```

### Manual Testing

```bash
# Test Standard RAG
python main.py --query "What is Python?"

# Test Self-RAG
python main.py --mode self-rag --query "What is Python?"

# Interactive mode
python main.py
```

## 🚨 Troubleshooting

### Common Issues

**1. API Key Errors**
```
Error: Invalid API key
```
→ Check `.env` file has correct `OPENAI_API_KEY` or `GOOGLE_API_KEY`

**2. No Retrieved Documents**
```
Warning: No documents retrieved
```
→ Ensure Qdrant is running: `docker-compose up -d`  
→ Check collection exists and has documents

**3. Import Errors**
```
ImportError: cannot import name 'graph'
```
→ Use correct imports:
- `from agent.graph_refined import graph` (Standard)
- `from agent.graph_self_rag import graph` (Self-RAG)

**4. Slow Performance**
→ Reduce `top_k` in `config.yml` retriever settings  
→ Disable reranking if not needed  
→ Use smaller LLM model

## 🎯 Choosing a Mode

### Standard RAG
**Use when**:
- Speed is important
- Queries are straightforward
- Hallucinations are less critical
- Resources are limited

**Features**:
- Single-pass generation
- Optional retrieval (router decides)
- Faster execution
- Lower API costs

### Self-RAG
**Use when**:
- Accuracy is critical
- Complex/technical queries
- Hallucination prevention needed
- Multiple refinements acceptable

**Features**:
- Iterative refinement (up to 3 loops)
- Hallucination detection
- Always retrieves context
- Higher accuracy, slower execution

---

## 🔗 Related Documentation

- **[Main README](../readme.md)**: System overview and quick start
- **[Retrievers](../retrievers/README.md)**: Search strategies
- **[Components](../components/README.md)**: Rerankers and filters
- **[Benchmarks](../benchmarks/README.md)**: Evaluation framework

---

**Built with LangGraph for flexible RAG workflows with single-pass and iterative refinement modes.**
