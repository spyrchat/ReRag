from table_pipeline.extractor import TableExtractor
from table_pipeline.mapper import SQLSchemaMapper
from table_pipeline.uploader import SQLUploader
from core.metadata import PageMetadata
import fitz


class TableRouter:
    def __init__(self):
        self.extractor = TableExtractor()
        self.mapper = SQLSchemaMapper()
        self.uploader = SQLUploader()

    def route(self, page: fitz.Page, metadata: PageMetadata):
        tables = self.extractor.extract(page)
        for table in tables:
            df, caption = self.mapper.map(table)
            self.uploader.upload_table(
                doc_id=metadata.doc_id,
                page=metadata.page,
                table_df=df,
                caption=caption
            )
