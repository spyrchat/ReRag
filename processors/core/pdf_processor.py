import fitz  # PyMuPDF
from processors.core.metadata import PageMetadata
from processors.table_pipeline.router import TableRouter
from processors.text_pipeline.router import TextRouter


class PDFProcessor:
    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.table_router = TableRouter()
        self.text_router = TextRouter()

    def process(self, filepath: str):
        pdf = fitz.open(filepath)

        for page_number, page in enumerate(pdf, start=1):
            metadata = PageMetadata(
                doc_id=self.doc_id,
                page=page_number,
                source=filepath.split("/")[-1]  # filename only
            )

            # Route text if it exists
            text = page.get_text("text")
            if text.strip():
                self.text_router.route(text, metadata)

            self.table_router.route(filepath, page_number, metadata)

        pdf.close()
        print(f"✓ Finished processing: {self.doc_id}")
