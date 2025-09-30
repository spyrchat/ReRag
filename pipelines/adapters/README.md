# StackOverflowAdapter and DatasetAdapter Usage Guide

This directory contains dataset adapters for ingestion and evaluation in the project. All dataset adapters should inherit from the `DatasetAdapter` interface (defined in `pipelines/contracts.py`).

## Abstract Interface: `DatasetAdapter`

All dataset adapters must implement the following interface:

```python
from pipelines.contracts import DatasetAdapter, DatasetSplit
from langchain_core.documents import Document
from typing import Iterable, List, Dict, Any

class MyDatasetAdapter(DatasetAdapter):
    def __init__(self, dataset_path: str, ...):
        ...

    @property
    def source_name(self) -> str:
        ...

    @property
    def version(self) -> str:
        ...

    def read_rows(self, split: DatasetSplit = DatasetSplit.ALL) -> Iterable[BaseRow]:
        """Yield dataset rows (questions, answers, etc.) as BaseRow or subclass."""
        ...

    def to_documents(self, rows: Iterable[BaseRow], split: DatasetSplit = DatasetSplit.ALL) -> List[Document]:
        """Convert rows to LangChain Documents for ingestion."""
        ...

    def get_evaluation_queries(self) -> List[Dict[str, Any]]:
        """Return a list of evaluation queries for benchmarking."""
        ...
```

- **`BaseRow`**: Define a row schema for your dataset (see `StackOverflowRow` for an example).
- **`read_rows`**: Should yield all relevant rows (questions, answers, etc.) for ingestion.
- **`to_documents`**: Should convert rows to `Document` objects, with all necessary metadata for retrieval and evaluation.
- **`get_evaluation_queries`**: Should return a list of queries (with expected document IDs) for benchmarking retrieval performance.

## Example: StackOverflowAdapter

See `stackoverflow.py` for a full implementation for the SOSum Stack Overflow dataset. This adapter:
- Reads questions and answers from CSV files.
- Converts only answers to retrievable documents, with question context in metadata.
- Provides evaluation queries that map questions and summaries to their corresponding answers.

## How to Add a New Adapter
1. **Inherit from `DatasetAdapter`** and implement all required methods.
2. **Define a row schema** (subclass of `BaseRow`) for your dataset.
3. **Implement ingestion logic** in `read_rows` and `to_documents`.
4. **Implement evaluation logic** in `get_evaluation_queries`.
5. **Test your adapter** with the project's ingestion and benchmarking scripts.

---

For more details, see the docstrings in `DatasetAdapter` and the example in `stackoverflow.py`.
