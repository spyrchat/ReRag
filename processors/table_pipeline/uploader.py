import json
import pandas as pd
from database.postgres_controller import PostgresController


class SQLUploader:
    def __init__(self):
        self.db = PostgresController()

    def upload_table(self, doc_id: str, page: int, table_df: pd.DataFrame, caption: str = None):
        json_str = table_df.to_json(orient="split")
        self.db.insert_table_asset(doc_id, page_number=page, table_json=json_str, caption=caption)


