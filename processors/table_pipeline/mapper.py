import pandas as pd
from typing import List, Tuple, Optional


class SQLSchemaMapper:
    def map(self, table: List[List[str]], caption_from_page: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
        """
        Converts a list-of-lists table into a pandas DataFrame and returns the most suitable caption.
        Prefers a provided caption if available, otherwise infers it.
        """
        if not table or len(table) < 2:
            raise ValueError("Table must have header and at least one data row.")

        header = table[0]
        rows = table[1:]
        df = pd.DataFrame(rows, columns=header)

        if caption_from_page and caption_from_page.strip():
            return df, caption_from_page.strip()

        caption = self._infer_caption(df)
        return df, caption

    def _infer_caption(self, df: pd.DataFrame) -> str:
        """
        Uses heuristics to guess the caption. Can be extended with LLM fallback.
        """
        if "Date" in df.columns and "Amount" in df.columns:
            return "Financial Transactions"
        return "Extracted Table"
