from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class PageMetadata:
    doc_id: str
    page: int
    source: Optional[str] = None
    language: Optional[str] = None
    section: Optional[str] = None

    def to_dict(self) -> dict:
        """
        Convert to a dict, omitting None values.
        """
        return {k: v for k, v in asdict(self).items() if v is not None}
