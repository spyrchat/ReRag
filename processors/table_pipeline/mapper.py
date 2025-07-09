import pandas as pd
from typing import List, Tuple, Optional
import logging

from logs.utils.logger import get_logger
logger = get_logger(__name__)


class SQLSchemaMapper:
    """
    Maps extracted table data (list of lists) to a Pandas DataFrame and infers or assigns a caption.
    """

    def map(self, table: List[List[str]], caption_from_page: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
        """
        Converts a 2D list table into a pandas DataFrame and determines the best caption.

        Args:
            table (List[List[str]]): The table, with first row as header.
            caption_from_page (Optional[str]): Optional extracted caption for the table.

        Returns:
            Tuple[pd.DataFrame, str]: DataFrame of the table, and a caption string.

        Raises:
            ValueError: If the table is too small or malformed.
        """
        if not table or len(table) < 2:
            logger.error("Table must have header and at least one data row.")
            raise ValueError(
                "Table must have header and at least one data row.")

        header = table[0]
        rows = table[1:]
        df = pd.DataFrame(rows, columns=header)
        logger.debug(f"Constructed DataFrame with shape {df.shape}")

        if caption_from_page and caption_from_page.strip():
            logger.info(
                f"Using provided caption: '{caption_from_page.strip()}'")
            return df, caption_from_page.strip()

        caption = self._infer_caption(df)
        logger.info(f"Inferred caption: '{caption}'")
        return df, caption

    def _infer_caption(self, df: pd.DataFrame) -> str:
        """
        Uses simple heuristics to generate a caption for the DataFrame.
        Extend this with LLM/AI for smarter inference if needed.

        Args:
            df (pd.DataFrame): DataFrame for which to infer a caption.

        Returns:
            str: Inferred caption string.
        """
        if "Date" in df.columns and "Amount" in df.columns:
            return "Financial Transactions"
        return "Extracted Table"
