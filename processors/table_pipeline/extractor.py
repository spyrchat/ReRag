import camelot
from typing import List, Tuple


class TableExtractor:
    def extract(self, filepath: str, page_number: int, verbose: bool = False) -> List[Tuple[List[List[str]], str]]:
        page_str = str(page_number)

        if verbose:
            print(f"Trying table extraction on page {page_number} with lattice...")

        try:
            tables = camelot.read_pdf(filepath, pages=page_str, flavor="lattice")
            if tables.n == 0:
                raise ValueError("No tables found with lattice.")
            if verbose:
                print(f"Found {tables.n} table(s) with lattice.")
        except Exception as e:
            if verbose:
                print(f"Lattice failed: {e}\nFalling back to stream...")
            try:
                tables = camelot.read_pdf(filepath, pages=page_str, flavor="stream")
                if verbose:
                    print(f"Found {tables.n} table(s) with stream.")
            except Exception as e:
                if verbose:
                    print(f"Stream failed: {e}")
                return []

        extracted = []
        for table in tables:
            data = table.df.values.tolist()
            extracted.append((data, None))  # No caption for now

        return extracted
