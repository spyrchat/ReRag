import camelot
from typing import List, Tuple
import logging

from logs.utils.logger import get_logger
logger = get_logger(__name__)


class TableExtractor:
    """
    Extracts tables from a PDF file using Camelot, trying lattice first and then falling back to stream.
    Returns a list of (table_data, caption) tuples for a given page.
    """

    def extract(self, filepath: str, page_number: int, verbose: bool = False) -> List[Tuple[List[List[str]], str]]:
        """
        Extract tables from a given PDF page using Camelot.
        Tries 'lattice' first; falls back to 'stream' if lattice fails.

        Args:
            filepath (str): Path to the PDF file.
            page_number (int): Page number to extract from.
            verbose (bool): If True, print step-by-step progress.

        Returns:
            List[Tuple[List[List[str]], str]]: List of (table data, caption) tuples.
        """
        page_str = str(page_number)

        if verbose:
            print(
                f"Trying table extraction on page {page_number} with lattice...")

        try:
            tables = camelot.read_pdf(
                filepath, pages=page_str, flavor="lattice")
            if tables.n == 0:
                raise ValueError("No tables found with lattice.")
            if verbose:
                print(f"Found {tables.n} table(s) with lattice.")
            logger.info(
                f"Found {tables.n} tables with lattice on page {page_number}.")
        except Exception as e:
            if verbose:
                print(f"Lattice failed: {e}\nFalling back to stream...")
            logger.warning(
                f"Lattice extraction failed for page {page_number}: {e}")
            try:
                tables = camelot.read_pdf(
                    filepath, pages=page_str, flavor="stream")
                if tables.n == 0:
                    if verbose:
                        print("No tables found with stream either.")
                    logger.info(
                        f"No tables found with stream on page {page_number}.")
                    return []
                if verbose:
                    print(f"Found {tables.n} table(s) with stream.")
                logger.info(
                    f"Found {tables.n} tables with stream on page {page_number}.")
            except Exception as e:
                if verbose:
                    print(f"Stream failed: {e}")
                logger.error(
                    f"Stream extraction failed for page {page_number}: {e}")
                return []

        extracted = []
        for table in tables:
            data = table.df.values.tolist()
            extracted.append((data, None))  # No caption for now

        return extracted
