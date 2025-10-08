# Configuration Management

Simple YAML-based configuration system for loading and managing application settings.

## 🎯 Overview

The config module provides:
- **YAML Configuration Loading**: Load settings from YAML files
- **Dictionary Access**: Standard Python dictionary access to configuration
- **Deep Merging**: Merge configuration overrides
- **Helper Functions**: Extract specific configuration sections

## 📁 Module Structure

```
config/
├── 📖 README.md                    # This file
├── ⚙️ config_loader.py             # Configuration loading functions
└── 📊 __init__.py                  # Module initialization
```

## ⚙️ Configuration Functions

### Available Functions

1. **`load_config(config_path)`** - Load YAML configuration file
2. **`get_retriever_config(config, retriever_type)`** - Extract retriever-specific config
3. **`get_benchmark_config(config)`** - Extract benchmark config with defaults
4. **`get_pipeline_config(config)`** - Extract pipeline config
5. **`load_config_with_overrides(config_path, overrides)`** - Load config with overrides

### Basic Usage
```python
from config.config_loader import load_config

# Load main configuration
config = load_config("config.yml")

# Access configuration values (standard dictionary access)
db_host = config['database']['host']
embedding_model = config['embedding']['model']
agent_temperature = config['agent']['temperature']
```

### Load with Overrides
```python
from config.config_loader import load_config_with_overrides

# Load config with overrides
overrides = {
    'database': {'host': 'production-qdrant.com'},
    'agent': {'temperature': 0.1}
}
config = load_config_with_overrides("config.yml", overrides=overrides)

print(config['database']['host'])  # 'production-qdrant.com'
print(config['agent']['temperature'])  # 0.1
```

## 📝 Configuration Structure

### Example Configuration (`config.yml`)
```yaml
# Database configuration
database:
  host: localhost
  port: 6333
  
# Embedding configuration
embedding:
  provider: google
  model: text-embedding-004
  strategy: hybrid
  
# Retrieval configuration
retrieval:
  top_k: 10
  score_threshold: 0.7
  
# Agent configuration (if using agent)
agent:
  llm_provider: openai
  model: gpt-4
  temperature: 0.1
```

Actual configuration structure depends on your specific setup. See `config.yml` in the project root for the complete configuration.

## 🔧 Advanced Usage

### Extract Retriever Configuration
```python
from config.config_loader import load_config, get_retriever_config

# Load main config
config = load_config("config.yml")

# Extract specific retriever config
dense_config = get_retriever_config(config, "dense")
hybrid_config = get_retriever_config(config, "hybrid")
```

### Extract Benchmark Configuration
```python
from config.config_loader import load_config, get_benchmark_config

config = load_config("config.yml")
benchmark_config = get_benchmark_config(config)

# Includes defaults for evaluation metrics
print(benchmark_config['evaluation']['k_values'])  # [1, 5, 10, 20]
print(benchmark_config['evaluation']['metrics'])   # ['precision', 'recall', ...]
```

### Extract Pipeline Configuration
```python
from config.config_loader import load_config, get_pipeline_config

config = load_config("config.yml")
pipeline_config = get_pipeline_config(config)

# Get default retriever
default_retriever = pipeline_config['default_retriever']  # 'hybrid'
```

### Deep Merge Configurations
```python
from config.config_loader import load_config_with_overrides

# Base configuration
base_config = load_config("config.yml")

# Override specific settings
overrides = {
    'retrieval': {
        'top_k': 20,
        'score_threshold': 0.8
    }
}

# Deep merge
final_config = load_config_with_overrides("config.yml", overrides=overrides)
```

## 🌍 Environment Variables

### Supported Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `ENVIRONMENT` | Deployment environment | `development` | `production` |
| `QDRANT_HOST` | Qdrant database host | `localhost` | `my-qdrant.com` |
| `QDRANT_PORT` | Qdrant database port | `6333` | `6333` |
| `QDRANT_API_KEY` | Qdrant API key | `""` | `your-api-key` |
| `GOOGLE_API_KEY` | Google AI API key | Required | `your-google-key` |
| `OPENAI_API_KEY` | OpenAI API key | Required | `sk-your-openai-key` |
| `VOYAGE_API_KEY` | Voyage AI API key | Optional | `your-voyage-key` |
| `EMBEDDING_STRATEGY` | Embedding strategy | `hybrid` | `dense`, `sparse`, `hybrid` |
| `LLM_PROVIDER` | LLM provider | `openai` | `openai`, `anthropic`, `google` |
| `LLM_MODEL` | LLM model | `gpt-4` | `gpt-4`, `claude-3-sonnet` |
| `LOG_LEVEL` | Logging level | `INFO` | `DEBUG`, `INFO`, `WARNING` |

### Environment File Example (`.env`)
```bash
# Database
QDRANT_HOST=my-qdrant-instance.com
QDRANT_API_KEY=your-qdrant-api-key

# Embedding providers
GOOGLE_API_KEY=your-google-api-key
OPENAI_API_KEY=sk-your-openai-api-key
VOYAGE_API_KEY=your-voyage-api-key

# System settings
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# Agent configuration
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.1

# Retrieval settings
EMBEDDING_STRATEGY=hybrid
HYBRID_ALPHA=0.7
RETRIEVAL_TOP_K=10

# Performance
MAX_WORKERS=8
EMBEDDING_BATCH_SIZE=64
```

##  Security Considerations

### Sensitive Data Handling
```python
import os
from config.config_loader import load_config

# Use environment variables for sensitive data
# Never hardcode API keys in config files
config = load_config("config.yml")

# Access from environment
api_key = os.getenv('QDRANT_API_KEY')
google_key = os.getenv('GOOGLE_API_KEY')
```

### Best Practices
- Never commit API keys or secrets to version control
- Use environment variables for sensitive data
- Use `.env` files locally (add to `.gitignore`)
- Use secure secret management in production (AWS Secrets Manager, etc.)

## 🧪 Testing Configuration

### Unit Testing
```python
import unittest
from config.config_loader import load_config

class TestConfiguration(unittest.TestCase):
    def test_config_loads(self):
        config = load_config("config.yml")
        self.assertIsNotNone(config)
        self.assertIn('database', config)
        self.assertIn('embedding', config)
    
    def test_config_with_overrides(self):
        from config.config_loader import load_config_with_overrides
        
        overrides = {'database': {'host': 'test-host'}}
        config = load_config_with_overrides("config.yml", overrides=overrides)
        self.assertEqual(config['database']['host'], 'test-host')
```

### Integration Testing
```python
from config.config_loader import load_config
from database.qdrant_controller import QdrantController

def test_config_integration():
    """Test configuration with actual components"""
    config = load_config("config.yml")
    
    # Test database connection with config
    db = QdrantController(
        host=config['database']['host'],
        port=config['database']['port']
    )
    assert db.health_check()
```

## 🐛 Troubleshooting

### Common Configuration Issues

**Configuration File Not Found:**
```python
from pathlib import Path
from config.config_loader import load_config

config_path = Path("config.yml")
if not config_path.exists():
    print(f"Config file not found: {config_path}")
else:
    config = load_config("config.yml")
```

**Invalid YAML Syntax:**
```python
import yaml

try:
    with open("config.yml") as f:
        config = yaml.safe_load(f)
except yaml.YAMLError as e:
    print(f"Invalid YAML: {e}")
```

**Missing Configuration Keys:**
```python
from config.config_loader import load_config

config = load_config("config.yml")

# Check for required keys
required_keys = ['database', 'embedding', 'retrieval']
missing = [key for key in required_keys if key not in config]
if missing:
    print(f"Missing required config keys: {missing}")
```

## 🎯 Best Practices

1. **Use YAML for configuration** - Human-readable and easy to edit
2. **Sensitive data in environment variables** - Never commit secrets
3. **Document your config structure** - Add comments in YAML files
4. **Test configuration loading** - Verify configs in your test suite
5. **Use overrides for environments** - Keep base config, override per environment
6. **Keep configs version controlled** - Track configuration changes

## 🔗 Integration

### With Main Application
```python
from config.config_loader import load_config
from database.qdrant_controller import QdrantController

# Load configuration at startup
config = load_config("config.yml")

# Pass configuration to components
db = QdrantController(
    host=config['database']['host'],
    port=config['database']['port']
)
```

### With Docker
```dockerfile
# Set environment variables in Docker
ENV QDRANT_HOST=qdrant
ENV GOOGLE_API_KEY=${GOOGLE_API_KEY}

# Copy configuration
COPY config.yml /app/config.yml
```

### Example Usage in Scripts
```python
from config.config_loader import load_config, get_retriever_config

# Load base config
config = load_config("config.yml")

# Get specific configurations
retriever_config = get_retriever_config(config, "hybrid")

# Use in your application
top_k = retriever_config['top_k']
alpha = retriever_config.get('alpha', 0.7)
```
