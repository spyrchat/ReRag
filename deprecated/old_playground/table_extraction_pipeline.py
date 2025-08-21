from processors.table_pipeline.extractor import TableExtractor
import os
import pandas as pd
import fitz  # PyMuPDF


def test_all_pdfs_in_sandbox():
    sandbox_dir = "papers"
    output_dir = "extraction_output"
    os.makedirs(output_dir, exist_ok=True)

    extractor = TableExtractor()
    pdf_files = [f for f in os.listdir(sandbox_dir) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in papers/")
        return

    for filename in pdf_files:
        filepath = os.path.join(sandbox_dir, filename)
        safe_filename = os.path.splitext(filename)[0].replace(" ", "_")

        try:
            pdf = fitz.open(filepath)
        except Exception as e:
            print(f"[{filename}] Failed to open: {e}")
            continue

        print(f"Testing {filename} with {len(pdf)} page(s)...")

        total_tables = 0
        for page_number in range(1, len(pdf) + 1):
            try:
                tables = extractor.extract(
                    filepath, page_number, verbose=False)
                table_count = len(tables)
                total_tables += table_count

                for i, (table_data, _) in enumerate(tables):
                    if not table_data or len(table_data) < 2:
                        continue  # skip empty or header-only tables

                    header = table_data[0]
                    rows = table_data[1:]
                    df = pd.DataFrame(rows, columns=header)

                    csv_path = os.path.join(
                        output_dir,
                        f"{safe_filename}_page{page_number}_table{i+1}.csv"
                    )
                    df.to_csv(csv_path, index=False)

                if table_count > 0:
                    print(f"  Page {page_number}: {table_count} table(s)")

            except Exception as e:
                print(f"  Page {page_number}: extraction failed: {e}")

        print(f"==> {filename}: total tables extracted = {total_tables}")
        pdf.close()


if __name__ == "__main__":
    test_all_pdfs_in_sandbox()
