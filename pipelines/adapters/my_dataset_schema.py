"""
Schema definitions for My Dataset.
"""
from typing import List, Optional
from pipelines.contracts import BaseRow


class MyDatasetRow(BaseRow):
    """Row schema for My Dataset."""
    external_id: str
    title: Optional[str] = None
    content: str
    category: Optional[str] = None
    tags: List[str] = []
    metadata: dict = {}

    # Add any dataset-specific fields here
    # author: Optional[str] = None
    # publication_date: Optional[str] = None
    # source_url: Optional[str] = None
