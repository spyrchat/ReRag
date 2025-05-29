# table_pipeline/extractor.py
import pdfplumber
import fitz
from typing import List


class TableExtractor:
    def extract(self, page: fitz.Page) -> List[List[List[str]]]:
        """
        Extracts tables as nested lists of cells using pdfplumber.
        Returns a list of tables, each table is a list of rows.
        """
        tables = []
        with pdfplumber.open(page.parent.name) as pdf:
            plumber_page = pdf.pages[page.number - 1]
            tables = plumber_page.extract_tables()
        return tables
