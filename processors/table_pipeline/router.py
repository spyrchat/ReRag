from .extractor import TableExtractor
from .mapper import SQLSchemaMapper
from .uploader import SQLUploader
from processors.core.metadata import PageMetadata


class TableRouter:
    def __init__(self):
        self.extractor = TableExtractor()
        self.mapper = SQLSchemaMapper()
        self.uploader = SQLUploader()

    def route(self, filepath: str, page_number: int, metadata: PageMetadata):
        tables = self.extractor.extract(filepath, page_number)

        if not tables:
            return  # skip if no tables on this page

        for table, _ in tables:
            df, caption = self.mapper.map(table)
            self.uploader.upload_table(
                doc_id=metadata.doc_id,
                page=metadata.page,
                table_df=df,
                caption=caption
            )
