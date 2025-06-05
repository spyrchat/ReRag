import logging
from .extractor import TableExtractor
from .mapper import SQLSchemaMapper
from .uploader import SQLUploader
from processors.core.metadata import PageMetadata

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TableRouter:
    def __init__(self):
        self.extractor = TableExtractor()
        self.mapper = SQLSchemaMapper()
        self.uploader = SQLUploader()

    def route(self, filepath: str, page_number: int, metadata: PageMetadata):
        tables = self.extractor.extract(filepath, page_number)

        if not tables:
            logger.info(f"No tables found on page {page_number} of {metadata.source}")
            return

        inserted = 0
        for i, (table, _) in enumerate(tables):
            try:
                df, caption = self.mapper.map(table)
                self.uploader.upload_table(
                    doc_id=metadata.doc_id,
                    page=metadata.page,
                    table_df=df,
                    caption=caption
                )
                inserted += 1
                logger.info(f"Inserted table {i+1} on page {metadata.page} of {metadata.source}")
            except ValueError as e:
                logger.warning(f"Skipping malformed table {i+1} on page {metadata.page} of {metadata.source}: {e}")

        logger.info(f"{inserted}/{len(tables)} tables inserted from page {metadata.page} of {metadata.source}")
