import os
import json
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings

# === Load environment ===
load_dotenv(override=True)

# === Config ===
INPUT_DIR = os.getenv("INPUT_DIR", "sandbox")
OUTPUT_JSON = os.getenv("OUTPUT_JSON", "synthetic_dataset.json")

# === Components ===
embedder = HuggingFaceEmbeddings()
splitter = SemanticChunker(
    embedder,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90
)

# === Dataset template accumulator ===
dataset = []

# === Process PDFs in sandbox ===
for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".pdf"):
        continue

    filepath = os.path.join(INPUT_DIR, filename)
    doc_id = os.path.splitext(filename)[0]
    pdf = fitz.open(filepath)

    for page_number, page in enumerate(pdf, start=1):
        text = page.get_text().strip()
        if not text:
            continue

        langchain_docs = splitter.create_documents([text])
        for chunk in langchain_docs:
            chunk.metadata = {"doc_id": doc_id, "page": page_number}
            dataset.append({
                "question": None,
                "answer": None,
                "context": chunk.page_content,
                "modality": "text",
                "source_meta": chunk.metadata
            })

# === Save dataset template ===
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"Created {len(dataset)} semantically chunked text samples.")
print(f"Template saved to {OUTPUT_JSON}")
