import logging
from .extractor import TableExtractor
from .mapper import SQLSchemaMapper
from .uploader import SQLUploader
from processors.core.metadata import PageMetadata
from logs.utils.logger import get_logger
logger = get_logger(__name__)
logger = logging.getLogger(__name__)


class TableRouter:
    """
    Orchestrates the extraction, mapping, and uploading of tables from a document page.
    Uses provided extractor, mapper, and uploader modules.
    """

    def __init__(self):
        """
        Initialize the TableRouter with extractor, mapper, and uploader components.
        """
        self.extractor = TableExtractor()
        self.mapper = SQLSchemaMapper()
        self.uploader = SQLUploader()
        logger.info(
            "TableRouter initialized with extractor, mapper, and uploader.")

    def route(self, filepath: str, page_number: int, metadata: PageMetadata):
        """
        Extract tables from a PDF page, map them to a DataFrame and caption, and upload to SQL.
        Args:
            filepath (str): Path to the PDF file.
            page_number (int): Page number to extract tables from.
            metadata (PageMetadata): Metadata object with document context.
        """
        tables = self.extractor.extract(filepath, page_number)

        if not tables:
            logger.info(
                f"No tables found on page {page_number} of {metadata.source}")
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
                logger.info(
                    f"Inserted table {i+1} on page {metadata.page} of {metadata.source}"
                )
            except ValueError as e:
                logger.warning(
                    f"Skipping malformed table {i+1} on page {metadata.page} of {metadata.source}: {e}"
                )

        logger.info(
            f"{inserted}/{len(tables)} tables inserted from page {metadata.page} of {metadata.source}"
        )
