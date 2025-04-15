import os
from typing import List, Dict
from langchain.schema import Document
from processors.pdf_processor import PDFProcessor
from processors.base import BaseProcessor


class ProcessorDispatcher:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.registry: Dict[str, BaseProcessor] = {
            ".pdf": PDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            # ".csv": CSVProcessor(...),
            # ".txt": TextProcessor(...),
        }

    def process_directory(self, root: str) -> List[Document]:
        for dirpath, _, filenames in os.walk(root):
            for file in filenames:
                ext = os.path.splitext(file)[1].lower()
                processor = self.registry.get(ext)
                if processor:
                    full_path = os.path.join(dirpath, file)
                    processor.add_file(full_path)

        all_docs = []
        for processor in self.registry.values():
            all_docs.extend(processor.process())
        return all_docs
