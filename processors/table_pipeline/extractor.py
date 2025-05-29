import camelot
from typing import List, Tuple


class TableExtractor:
    def extract(self, filepath: str, page_number: int) -> List[Tuple[List[List[str]], str]]:
        page_str = str(page_number)

        try:
            tables = camelot.read_pdf(filepath, pages=page_str, flavor="lattice")
            if tables.n == 0:
                raise ValueError("No tables found with lattice.")
        except Exception:
            tables = camelot.read_pdf(filepath, pages=page_str, flavor="stream")

        extracted = []
        for table in tables:
            data = table.df.values.tolist()
            extracted.append((data, None))  # Caption comes separately
        return extracted
