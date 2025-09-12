"""
Standalone SOSum ingestion script.
This script works without the full pipeline dependencies.
"""
import csv
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

def compute_hash(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]

def parse_list_field(field_value: str) -> List[str]:
    """Parse a field that might be a string representation of a list."""
    if not field_value:
        return []
    
    if field_value.startswith('[') and field_value.endswith(']'):
        try:
            import ast
            result = ast.literal_eval(field_value)
            if isinstance(result, list):
                return [str(item).strip() for item in result if str(item).strip()]
        except:
            # Fallback: strip brackets and split by comma
            return [item.strip().strip("'\"") for item in field_value.strip('[]').split(',') if item.strip()]
    
    return [field_value.strip()]

def process_sosum_dataset(dataset_path: str, output_dir: str = "output/sosum_processed"):
    """Process SOSum dataset and export as JSON."""
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find CSV files
    question_file = dataset_path / "question.csv"
    answer_file = dataset_path / "answer.csv"
    
    if not question_file.exists():
        data_dir = dataset_path / "data"
        if data_dir.exists():
            question_file = data_dir / "question.csv"
            answer_file = data_dir / "answer.csv"
    
    if not question_file.exists() or not answer_file.exists():
        raise FileNotFoundError(f"CSV files not found in {dataset_path}")
    
    print(f"📂 Processing SOSum dataset from: {dataset_path}")
    print(f"📄 Questions: {question_file}")
    print(f"📄 Answers: {answer_file}")
    
    documents = []
    evaluation_queries = []
    
    # Process questions
    print("\n🔍 Processing questions...")
    question_count = 0
    with open(question_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader):
            try:
                question_id = row.get("Question Id", f"q{row_num}")
                title = row.get("Question Title", "")
                body_raw = row.get("Question Body", "")
                tags_raw = row.get("Tags", "")
                question_type = row.get("Question Type", "")
                
                # Parse body (might be list of sentences)
                body_list = parse_list_field(body_raw)
                body = " ".join(body_list) if body_list else ""
                
                # Parse tags
                tags = parse_list_field(tags_raw) if tags_raw else []
                
                # Create document content
                content = f"Title: {title}\n\nQuestion: {body}" if title and body else (title or body)
                
                if content.strip():
                    doc_id = f"q_{question_id}"
                    content_hash = compute_hash(content)
                    
                    document = {
                        "id": doc_id,
                        "content": content,
                        "content_hash": content_hash,
                        "metadata": {
                            "external_id": doc_id,
                            "source": "stackoverflow_sosum",
                            "post_type": "question",
                            "title": title,
                            "tags": tags,
                            "question_type": int(question_type) if question_type.isdigit() else None,
                            "char_count": len(content),
                            "processed_at": datetime.now().isoformat()
                        }
                    }
                    documents.append(document)
                    question_count += 1
                    
                    # Create evaluation query from title
                    if title and len(title) > 10:
                        evaluation_queries.append({
                            "query_id": f"eval_q_{question_id}",
                            "query": title,
                            "expected_docs": [doc_id],
                            "query_type": "question_title"
                        })
                
            except Exception as e:
                print(f"Error processing question {row_num}: {e}")
                continue
    
    print(f"✅ Processed {question_count} questions")
    
    # Process answers
    print("\n🔍 Processing answers...")
    answer_count = 0
    with open(answer_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader):
            try:
                answer_id = row.get("Answer Id", f"a{row_num}")
                body_raw = row.get("Answer Body", "")
                summary_raw = row.get("Summary", "")
                
                # Parse body (might be list of sentences)
                body_list = parse_list_field(body_raw)
                body = " ".join(body_list) if body_list else ""
                
                # Parse summary (might be list of sentences)
                summary_list = parse_list_field(summary_raw) if summary_raw else []
                summary = " ".join(summary_list) if summary_list else ""
                
                # Create document content
                content = body
                if summary:
                    content = f"Answer: {body}\n\nSummary: {summary}"
                
                if content.strip():
                    doc_id = f"a_{answer_id}"
                    content_hash = compute_hash(content)
                    
                    document = {
                        "id": doc_id,
                        "content": content,
                        "content_hash": content_hash,
                        "metadata": {
                            "external_id": doc_id,
                            "source": "stackoverflow_sosum",
                            "post_type": "answer",
                            "has_summary": bool(summary),
                            "summary": summary if summary else None,
                            "char_count": len(content),
                            "processed_at": datetime.now().isoformat()
                        }
                    }
                    documents.append(document)
                    answer_count += 1
                    
                    # Create evaluation query from summary
                    if summary and len(summary) > 20:
                        # Use first sentence of summary as query
                        summary_sentences = summary.split('.')
                        query = summary_sentences[0].strip() + "." if summary_sentences else summary[:50]
                        
                        if len(query) > 10:
                            evaluation_queries.append({
                                "query_id": f"eval_a_{answer_id}",
                                "query": query,
                                "expected_docs": [doc_id],
                                "query_type": "answer_summary"
                            })
                
            except Exception as e:
                print(f"Error processing answer {row_num}: {e}")
                continue
    
    print(f"✅ Processed {answer_count} answers")
    
    # Save results
    documents_file = output_dir / "documents.json"
    queries_file = output_dir / "evaluation_queries.json"
    stats_file = output_dir / "stats.json"
    
    with open(documents_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    
    with open(queries_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_queries, f, indent=2, ensure_ascii=False)
    
    # Generate statistics
    stats = {
        "dataset_name": "SOSum Stack Overflow",
        "processed_at": datetime.now().isoformat(),
        "total_documents": len(documents),
        "questions": question_count,
        "answers": answer_count,
        "evaluation_queries": len(evaluation_queries),
        "files": {
            "documents": str(documents_file),
            "evaluation_queries": str(queries_file),
        },
        "content_stats": {
            "avg_char_count": sum(d["metadata"]["char_count"] for d in documents) / len(documents) if documents else 0,
            "min_char_count": min(d["metadata"]["char_count"] for d in documents) if documents else 0,
            "max_char_count": max(d["metadata"]["char_count"] for d in documents) if documents else 0,
        }
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Processing complete!")
    print(f"   Total documents: {len(documents)}")
    print(f"   Questions: {question_count}")
    print(f"   Answers: {answer_count}")
    print(f"   Evaluation queries: {len(evaluation_queries)}")
    print(f"   Avg content length: {stats['content_stats']['avg_char_count']:.0f} chars")
    print(f"\n💾 Files saved to: {output_dir}")
    print(f"   Documents: {documents_file}")
    print(f"   Queries: {queries_file}")
    print(f"   Stats: {stats_file}")
    
    return documents, evaluation_queries, stats

def main():
    """Main function."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python standalone_sosum_processor.py <dataset_path> [output_dir]")
        print("Example: python standalone_sosum_processor.py ../datasets/sosum")
        return 1
    
    dataset_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output/sosum_processed"
    
    try:
        documents, queries, stats = process_sosum_dataset(dataset_path, output_dir)
        
        print(f"\n🎯 Ready for ingestion!")
        print(f"   You now have {len(documents)} processed documents")
        print(f"   Each document has a unique ID and content hash for deduplication")
        print(f"   {len(queries)} evaluation queries are available for testing")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
