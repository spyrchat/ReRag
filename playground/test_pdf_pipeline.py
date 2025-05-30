import os
from processors.core.pdf_processor import PDFProcessor
import uuid


def run_all_pdfs_in_sandbox():
    sandbox_dir = "sandbox"
    files = [f for f in os.listdir(sandbox_dir) if f.endswith(".pdf")]

    if not files:
        print("No PDFs found in sandbox/")
        return

    for filename in files:
        path = os.path.join(sandbox_dir, filename)
        doc_id = str(uuid.uuid4())  # unique ID for each document

        print(f"Processing: {filename} → doc_id={doc_id}")
        processor = PDFProcessor(doc_id=doc_id)
        processor.process(path)

    print("All files processed.")


if __name__ == "__main__":
    run_all_pdfs_in_sandbox()
