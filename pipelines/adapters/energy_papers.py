"""
Adapter for energy research papers (PDF documents).
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Iterable

from pipelines.contracts import BaseRow, DatasetAdapter, DatasetSplit
from langchain_core.documents import Document


class EnergyPaperRow(BaseRow):
    """Row schema for energy research papers."""
    title: str
    file_path: str
    content: str = ""
    authors: List[str] = []
    abstract: str = ""
    keywords: List[str] = []
    year: int = 0
    
    class Config:
        extra = "allow"


class EnergyPapersAdapter(DatasetAdapter):
    """Adapter for energy research papers dataset."""
    
    def __init__(self, papers_path: str, version: str = "1.0.0"):
        self.papers_path = Path(papers_path)
        self._version = version
        
        if not self.papers_path.exists():
            raise FileNotFoundError(f"Papers directory not found at {self.papers_path}")
    
    @property
    def source_name(self) -> str:
        return "energy_papers"
    
    @property
    def version(self) -> str:
        return self._version
    
    def read_rows(self, split: DatasetSplit = DatasetSplit.ALL) -> Iterable[EnergyPaperRow]:
        """Read PDF files from papers directory."""
        pdf_files = list(self.papers_path.glob("*.pdf"))
        
        # Simple split logic based on filename patterns or random split
        total_files = len(pdf_files)
        if split == DatasetSplit.TRAIN:
            pdf_files = pdf_files[:int(0.7 * total_files)]
        elif split == DatasetSplit.VALIDATION:
            pdf_files = pdf_files[int(0.7 * total_files):int(0.85 * total_files)]
        elif split == DatasetSplit.TEST:
            pdf_files = pdf_files[int(0.85 * total_files):]
        
        for pdf_path in pdf_files:
            try:
                yield self._extract_paper_info(pdf_path)
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
                continue
    
    def _extract_paper_info(self, pdf_path: Path) -> EnergyPaperRow:
        """Extract basic information from PDF file."""
        # Extract title from filename (clean it up)
        title = pdf_path.stem
        title = title.replace("_", " ").replace("-", " ")
        # Remove common patterns like "v1", "v2", etc.
        import re
        title = re.sub(r'\s+v\d+.*$', '', title)
        title = title.strip()
        
        # Try to extract more metadata if available
        # For now, use basic file-based extraction
        # In a real implementation, you'd use PyMuPDF, pdfplumber, etc.
        
        content = ""
        authors = []
        abstract = ""
        keywords = []
        year = 0
        
        # Extract year from filename if present
        year_match = re.search(r'20\d{2}', pdf_path.name)
        if year_match:
            year = int(year_match.group())
        
        # For this example, we'll simulate content extraction
        # In practice, you'd integrate with your existing PDF processing
        try:
            # Placeholder for actual PDF text extraction
            # You could integrate with your existing PDF processors here
            content = f"Content from {pdf_path.name} would be extracted here"
        except Exception as e:
            print(f"Could not extract content from {pdf_path}: {e}")
        
        return EnergyPaperRow(
            external_id=pdf_path.stem,
            title=title,
            file_path=str(pdf_path),
            content=content,
            authors=authors,
            abstract=abstract,
            keywords=keywords,
            year=year
        )
    
    def to_documents(self, rows: List[EnergyPaperRow], split: DatasetSplit) -> List[Document]:
        """Convert paper rows to Documents."""
        documents = []
        
        for row in rows:
            # Create document content
            content_parts = []
            if row.title:
                content_parts.append(f"Title: {row.title}")
            if row.abstract:
                content_parts.append(f"Abstract: {row.abstract}")
            if row.content:
                content_parts.append(row.content)
            
            full_text = "\n\n".join(content_parts)
            if not full_text.strip():
                continue
            
            metadata = {
                "external_id": row.external_id,
                "title": row.title,
                "file_path": row.file_path,
                "authors": row.authors,
                "keywords": row.keywords,
                "year": row.year,
                "source": self.source_name,
                "dataset_version": self.version,
                "split": split.value,
                "doc_type": "research_paper"
            }
            
            documents.append(Document(
                page_content=full_text,
                metadata=metadata
            ))
        
        return documents
    
    def get_evaluation_queries(self, split: DatasetSplit = DatasetSplit.TEST) -> List[Dict[str, Any]]:
        """Return evaluation queries for energy papers."""
        eval_queries = []
        
        # Generate queries from paper titles and abstracts
        common_energy_queries = [
            "renewable energy optimization",
            "solar panel efficiency",
            "wind turbine design",
            "energy storage systems",
            "smart grid technology",
            "carbon emission reduction",
            "energy management systems",
            "power system reliability",
            "sustainable energy development",
            "energy efficiency improvements"
        ]
        
        for i, query in enumerate(common_energy_queries):
            # For energy papers, relevance would need to be determined
            # by semantic similarity or keyword matching
            eval_queries.append({
                "query_id": f"energy_query_{i}",
                "query": query,
                "relevant_doc_ids": [],  # Would need manual annotation
                "domain": "energy"
            })
        
        return eval_queries
