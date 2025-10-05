# Dynamic Adapter System - Quick Reference

## Overview

The ingestion pipeline now supports **dynamic adapter loading**, allowing you to add new dataset adapters without modifying any core code. Just create your adapter class and reference it by name!

## Key Benefits

✅ **Zero Code Changes**: Add adapters without touching `bin/ingest.py` or any core files  
✅ **Scalable**: Add unlimited adapters by creating new classes  
✅ **Flexible**: Use shortcuts or full class paths  
✅ **Extensible**: Anyone can create custom adapters independently  
✅ **Version Control Friendly**: No merge conflicts in core files  

## Quick Start

### 1. Use Built-in Adapters (No Setup Required)

```bash
# Use built-in shortcuts
python bin/ingest.py ingest stackoverflow /path/to/data
python bin/ingest.py ingest natural_questions /path/to/data
python bin/ingest.py ingest energy_papers /path/to/data
```

### 2. Create Custom Adapter (3 Steps)

**Step 1: Create your adapter class**
```python
# my_package/adapters/my_adapter.py
from pipelines.contracts import DatasetAdapter, BaseRow, DatasetSplit

class MyAdapter(DatasetAdapter):
    def __init__(self, dataset_path: str, version: str = "1.0.0", **kwargs):
        self.dataset_path = dataset_path
        self._version = version
    
    @property
    def source_name(self) -> str:
        return "my_dataset"
    
    @property
    def version(self) -> str:
        return self._version
    
    def read_rows(self, split): ...
    def to_documents(self, rows, split): ...
    def get_evaluation_queries(self, split): ...
```

**Step 2: Use it immediately (no code changes!)**
```bash
# Use full class path - works immediately!
python bin/ingest.py ingest \
  "my_package.adapters.my_adapter.MyAdapter" \
  /path/to/data \
  --config my_config.yml
```

**Step 3 (Optional): Register a shortcut**
```python
# In pipelines/adapters/loader.py (one-time setup)
ADAPTER_SHORTCUTS = {
    # ... existing shortcuts ...
    "my_adapter": "my_package.adapters.my_adapter.MyAdapter",
}
```

Now you can use: `python bin/ingest.py ingest my_adapter /path/to/data`

## Built-in Adapter Shortcuts

| Shortcut | Description |
|----------|-------------|
| `stackoverflow` | Stack Overflow Q&A (SOSum format) |
| `natural_questions` | Google Natural Questions dataset |
| `energy_papers` | Energy research papers (PDFs) |
| `beir` | BEIR benchmark datasets |

## Usage Examples

### Example 1: Built-in Adapter
```bash
python bin/ingest.py ingest stackoverflow /data/sosum/ --config config.yml
```

### Example 2: Custom Adapter (Full Path)
```bash
# No code changes needed!
python bin/ingest.py ingest \
  "company.adapters.ProductCatalogAdapter" \
  /data/products/ \
  --config products.yml
```

### Example 3: Adapter with Custom Parameters

**Config file:**
```yaml
dataset:
  adapter: "my_package.APIAdapter"
  adapter_kwargs:
    api_key_env: "MY_API_KEY"
    rate_limit: 100
```

**Adapter:**
```python
class APIAdapter(DatasetAdapter):
    def __init__(self, dataset_path: str, version: str = "1.0.0", **kwargs):
        self.api_key = os.getenv(kwargs.get("api_key_env"))
        self.rate_limit = kwargs.get("rate_limit", 10)
```

### Example 4: Programmatic Usage

```python
from pipelines.adapters.loader import AdapterLoader

# Load by shortcut
adapter = AdapterLoader.load_adapter("stackoverflow", "/data")

# Load by full path
adapter = AdapterLoader.load_adapter(
    "my_pkg.adapters.MyAdapter",
    "/data",
    custom_param="value"
)

# Load from config
from config.config_loader import load_config
config = load_config("my_config.yml")
adapter = AdapterLoader.load_from_config(config)
```

## Migration from Hardcoded Adapters

**Before (required code changes):**
```python
# bin/ingest.py - had to modify this file!
adapters = {
    "stackoverflow": StackOverflowAdapter,
    "my_adapter": MyAdapter,  # Manual registration
}
```

**After (no code changes):**
```bash
# Just use the adapter directly
python bin/ingest.py ingest \
  "my_package.adapters.MyAdapter" \
  /path/to/data
```

## Files Changed

- `pipelines/adapters/loader.py` - New dynamic loader utility
- `bin/ingest.py` - Updated to use dynamic loading
- `pipelines/configs/datasets/*.yml` - Updated config format
- `docs/DYNAMIC_ADAPTERS.md` - Comprehensive guide
- `examples/custom_adapter_example.py` - Working example

## Documentation

- **Full Guide**: `docs/DYNAMIC_ADAPTERS.md`
- **Example Code**: `examples/custom_adapter_example.py`
- **Adapter Template**: `pipelines/adapters/README.md`

## See Also

- Adapter Interface: `pipelines/contracts.py`
- Built-in Adapters: `pipelines/adapters/`
- Configuration: `pipelines/configs/datasets/`
