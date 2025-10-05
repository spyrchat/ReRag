# Adding Custom Adapters Without Code Changes

This guide shows you how to add new dataset adapters to the ingestion pipeline **without modifying any core code**. Just create your adapter class and reference it in your config!

## Quick Start: Add a New Adapter in 3 Steps

### Step 1: Create Your Adapter Class

Create a new file in `pipelines/adapters/` (or anywhere in your Python path):

```python
# my_project/adapters/my_custom_adapter.py
from typing import List, Dict, Any, Iterable
from pathlib import Path
from pipelines.contracts import DatasetAdapter, BaseRow, DatasetSplit
from langchain_core.documents import Document


class MyCustomRow(BaseRow):
    """Row schema for your dataset."""
    title: str
    content: str
    # Add your custom fields...


class MyCustomAdapter(DatasetAdapter):
    """Adapter for my custom dataset."""
    
    def __init__(self, dataset_path: str, version: str = "1.0.0", **kwargs):
        """
        Initialize adapter.
        
        Args:
            dataset_path: Path to your dataset files
            version: Dataset version
            **kwargs: Any custom parameters you need
        """
        self.dataset_path = Path(dataset_path)
        self._version = version
        # Use kwargs for custom parameters
        self.custom_param = kwargs.get("custom_param", "default")
    
    @property
    def source_name(self) -> str:
        return "my_custom_dataset"
    
    @property
    def version(self) -> str:
        return self._version
    
    def read_rows(self, split: DatasetSplit = DatasetSplit.ALL) -> Iterable[MyCustomRow]:
        """Read your dataset files and yield rows."""
        # Your data loading logic here
        for file in self.dataset_path.glob("*.jsonl"):
            # Example: yield rows from your data
            pass
    
    def to_documents(self, rows: List[MyCustomRow], split: DatasetSplit) -> List[Document]:
        """Convert rows to LangChain documents."""
        documents = []
        for row in rows:
            doc = Document(
                page_content=f"{row.title}\n\n{row.content}",
                metadata={
                    "source": self.source_name,
                    "external_id": row.external_id,
                    "split": split.value,
                    "dataset_version": self.version
                }
            )
            documents.append(doc)
        return documents
    
    def get_evaluation_queries(self, split: DatasetSplit = DatasetSplit.TEST) -> List[Dict[str, Any]]:
        """Return evaluation queries (optional)."""
        return []
```

### Step 2: Create a Config File

Create a YAML config that references your adapter:

```yaml
# pipelines/configs/datasets/my_custom_dataset.yml
dataset:
  name: "my_custom_dataset"
  version: "1.0.0"
  
  # Option A: Use full class path (no code changes needed!)
  adapter: "my_project.adapters.my_custom_adapter.MyCustomAdapter"
  
  # Option B: Register a shortcut (see Step 3)
  # adapter: "my_custom"
  
  # Optional: Path to dataset (can override via CLI)
  path: "/path/to/my/data"
  
  # Optional: Pass custom parameters to adapter constructor
  adapter_kwargs:
    custom_param: "my_value"
    another_param: 123

# Standard pipeline configuration
chunking:
  strategy: "recursive"
  chunk_size: 500
  chunk_overlap: 50

embedding:
  strategy: "dense"
  dense:
    provider: "hf"
    model: "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: 32

qdrant:
  collection: "my_custom_dataset_v1"
  dense_vector_name: "dense"
```

### Step 3: Run Ingestion

That's it! No code changes needed. Just run:

```bash
# Using full class path
python bin/ingest.py ingest \
  "my_project.adapters.my_custom_adapter.MyCustomAdapter" \
  /path/to/data \
  --config pipelines/configs/datasets/my_custom_dataset.yml

# Or if you registered a shortcut (see below)
python bin/ingest.py ingest my_custom /path/to/data \
  --config pipelines/configs/datasets/my_custom_dataset.yml
```

## Advanced: Register Custom Shortcuts

If you don't want to type the full class path every time, register a shortcut:

### Option 1: In Your Code (Before Running Pipeline)

```python
from pipelines.adapters.loader import AdapterLoader

# Register your custom adapter
AdapterLoader.register_shortcut(
    "my_custom",
    "my_project.adapters.my_custom_adapter.MyCustomAdapter"
)

# Now you can use the shortcut
adapter = AdapterLoader.load_adapter("my_custom", "/path/to/data")
```

### Option 2: Modify loader.py Once (Optional)

Edit `pipelines/adapters/loader.py` and add to `ADAPTER_SHORTCUTS`:

```python
ADAPTER_SHORTCUTS = {
    "stackoverflow": "pipelines.adapters.stackoverflow.StackOverflowAdapter",
    "natural_questions": "pipelines.adapters.natural_questions.NaturalQuestionsAdapter",
    "energy_papers": "pipelines.adapters.energy_papers.EnergyPapersAdapter",
    "beir": "pipelines.adapters.beir_base.GenericBeirAdapter",
    # Add your custom adapters here
    "my_custom": "my_project.adapters.my_custom_adapter.MyCustomAdapter",
}
```

## Built-in Adapter Shortcuts

These adapters are available by shortcut:

| Shortcut | Full Path | Description |
|----------|-----------|-------------|
| `stackoverflow` | `pipelines.adapters.stackoverflow.StackOverflowAdapter` | Stack Overflow Q&A |
| `natural_questions` | `pipelines.adapters.natural_questions.NaturalQuestionsAdapter` | Google Natural Questions |
| `energy_papers` | `pipelines.adapters.energy_papers.EnergyPapersAdapter` | Research papers (PDFs) |
| `beir` | `pipelines.adapters.beir_base.GenericBeirAdapter` | BEIR benchmark datasets |

## Usage Examples

### Example 1: Use Built-in Adapter

```bash
# Using shortcut
python bin/ingest.py ingest stackoverflow /path/to/sosum/data

# Same as above, but explicit
python bin/ingest.py ingest \
  "pipelines.adapters.stackoverflow.StackOverflowAdapter" \
  /path/to/sosum/data
```

### Example 2: Use Custom Adapter (No Code Changes!)

```bash
# Your custom adapter in a different package
python bin/ingest.py ingest \
  "my_company.data_adapters.ProductCatalogAdapter" \
  /data/products/ \
  --config configs/products.yml
```

### Example 3: Adapter with Custom Parameters

Your config:
```yaml
dataset:
  adapter: "my_pkg.adapters.APIAdapter"
  adapter_kwargs:
    api_key_env: "MY_API_KEY"
    rate_limit: 100
    batch_size: 50
```

Your adapter:
```python
class APIAdapter(DatasetAdapter):
    def __init__(self, dataset_path: str, version: str = "1.0.0", **kwargs):
        self.api_key = os.getenv(kwargs.get("api_key_env", "API_KEY"))
        self.rate_limit = kwargs.get("rate_limit", 10)
        self.batch_size = kwargs.get("batch_size", 32)
        # ...
```

### Example 4: Load Adapter Programmatically

```python
from pipelines.adapters.loader import AdapterLoader

# Load by shortcut
adapter = AdapterLoader.load_adapter(
    "stackoverflow",
    "/path/to/data",
    version="1.0.0"
)

# Load by full path
adapter = AdapterLoader.load_adapter(
    "my_project.adapters.CustomAdapter",
    "/path/to/data",
    custom_param="value"
)

# Load from config file
from config.config_loader import load_config

config = load_config("my_config.yml")
adapter = AdapterLoader.load_from_config(
    config,
    dataset_path="/override/path"  # Optional override
)
```

## Adapter Requirements

Your custom adapter must:

1. ✅ Inherit from `DatasetAdapter` (defined in `pipelines/contracts.py`)
2. ✅ Implement required methods:
   - `source_name` property
   - `version` property
   - `read_rows(split)` method
   - `to_documents(rows, split)` method
   - `get_evaluation_queries(split)` method (can return empty list)
3. ✅ Accept `(dataset_path, version, **kwargs)` in constructor
4. ✅ Be importable (in Python path or installed package)

## Benefits of Dynamic Loading

- ✅ **No code modifications**: Add adapters without touching core code
- ✅ **Version control friendly**: Adapters in separate files/packages
- ✅ **Extensible**: Anyone can add adapters without merge conflicts
- ✅ **Testable**: Each adapter is independently testable
- ✅ **Reusable**: Share adapters across projects
- ✅ **Type-safe**: Full IDE support and type checking

## Troubleshooting

### Error: "Could not import adapter module"

**Problem**: Python can't find your adapter module.

**Solution**: Make sure your adapter is:
- In a directory that's in `PYTHONPATH`, or
- Installed as a package (`pip install -e .`), or
- In the `pipelines/adapters/` directory

```bash
# Add to PYTHONATH temporarily
export PYTHONPATH="${PYTHONPATH}:/path/to/your/code"

# Or install your package
cd /path/to/your/package
pip install -e .
```

### Error: "Module does not have class"

**Problem**: The class name doesn't match what's in the module.

**Solution**: Check the exact class name in your adapter file. Python is case-sensitive!

```bash
# Check what's actually in the module
python -c "import my_module; print(dir(my_module))"
```

### Error: "Failed to instantiate adapter"

**Problem**: Your adapter's `__init__` method signature is incompatible.

**Solution**: Make sure your adapter accepts:
```python
def __init__(self, dataset_path: str, version: str = "1.0.0", **kwargs):
    # Your code here
```

## Migration Guide: From Hardcoded to Dynamic

If you have existing code that hardcodes adapters, here's how to migrate:

**Before (hardcoded):**
```python
# bin/ingest.py
adapters = {
    "stackoverflow": StackOverflowAdapter,
    "my_adapter": MyAdapter,  # Have to modify code!
}
```

**After (dynamic):**
```bash
# Just use the adapter directly
python bin/ingest.py ingest \
  "my_package.adapters.MyAdapter" \
  /path/to/data
```

Or register a shortcut once in `loader.py` and never modify CLI code again!

## See Also

- `pipelines/adapters/stackoverflow.py` - Example adapter implementation
- `pipelines/contracts.py` - DatasetAdapter interface definition
- `pipelines/adapters/loader.py` - Dynamic loading implementation
- `docs/MLOPS_PIPELINE_ARCHITECTURE.md` - Overall pipeline architecture
