# Benchmark Dynamic Adapter Loading - Fix Summary

## Problem

When implementing dynamic adapter loading for benchmarks, we encountered an error:

```
StackOverflowBenchmarkAdapter.__init__() got multiple values for argument 'qdrant_client'
```

This occurred because:
1. **Ingestion adapters** have signature: `__init__(dataset_path, version, **kwargs)`
2. **Benchmark adapters** had signature: `__init__(dataset_path, qdrant_client=None, collection_name=None)`

The `AdapterLoader` was designed for ingestion adapters and calls:
```python
adapter_class(dataset_path, version, **kwargs)
```

When `qdrant_client` was passed in `**kwargs`, it conflicted with the benchmark adapter's explicit parameter.

## Solution

### 1. Updated Benchmark Adapter Signature

Made `StackOverflowBenchmarkAdapter` compatible with `AdapterLoader`:

```python
# OLD
def __init__(self, dataset_path: str, qdrant_client=None, collection_name=None):

# NEW
def __init__(self, dataset_path: str, version: str = "1.0.0", 
             qdrant_client=None, collection_name=None, **kwargs):
```

**Changes:**
- Added `version` parameter (for compatibility with `AdapterLoader`)
- Added `**kwargs` to absorb any extra arguments
- Stored `version` attribute for compatibility

### 2. Updated AdapterLoader Logging

Made `AdapterLoader` more flexible for different adapter types:

```python
# OLD - Assumed all adapters have source_name
logger.info(f"Successfully loaded adapter: {adapter.source_name} v{adapter.version}")

# NEW - Falls back to name or class_name
adapter_name = getattr(adapter, 'source_name', None) or getattr(adapter, 'name', class_name)
adapter_version = getattr(adapter, 'version', version)
logger.info(f"Successfully loaded adapter: {adapter_name} v{adapter_version}")
```

## Files Modified

1. **`benchmarks/benchmarks_adapters.py`**
   - Updated `StackOverflowBenchmarkAdapter.__init__()` signature
   - Added docstring explaining parameters
   - Added `self.version` attribute

2. **`pipelines/adapters/loader.py`**
   - Made logging more flexible for different adapter types
   - Uses `getattr()` with fallbacks for `source_name` and `version`

## Verification

Tested with:
```bash
python benchmarks/experiment1.py --test
```

**Result:** ✅ Success
- Adapter loads correctly from YAML config
- Qdrant client and collection_name passed properly
- Benchmark runs successfully with 10 test queries

## Key Takeaway

When designing dynamic loading systems that support multiple adapter types:
1. **Standardize signatures** where possible
2. **Use flexible parameters** (`**kwargs`) to absorb extra arguments
3. **Make loaders resilient** with fallback attribute access
4. **Document expected signatures** clearly

## Adapter Compatibility Matrix

| Adapter Type | Required Parameters | Optional Parameters |
|--------------|---------------------|---------------------|
| **Ingestion** | `dataset_path`, `version` | Any via `**kwargs` |
| **Benchmark** | `dataset_path`, `version` | `qdrant_client`, `collection_name`, `**kwargs` |

Both now work with the unified `AdapterLoader`!

## Related Documentation

- [Dynamic Adapter Loading for Benchmarks](./DYNAMIC_ADAPTER_BENCHMARKS.md)
- [Dynamic Adapter Loading for Ingestion](./docs/DYNAMIC_ADAPTERS.md)
