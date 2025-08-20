#!/usr/bin/env python3
"""
Ingestion pipeline CLI - single entrypoint for all ingestion operations.
Implements canary -> verify -> promote workflow with comprehensive logging.
"""
import argparse
import logging
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.ingest.pipeline import IngestionPipeline, BatchIngestionPipeline
from pipelines.eval.evaluator import RetrievalEvaluator
from pipelines.adapters.natural_questions import NaturalQuestionsAdapter
from pipelines.adapters.stackoverflow import StackOverflowAdapter
from pipelines.adapters.energy_papers import EnergyPapersAdapter
from pipelines.contracts import DatasetSplit
from config.config_loader import load_config


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logs_dir / "ingestion.log")
        ]
    )


def get_adapter(adapter_type: str, dataset_path: str, version: str = "1.0.0"):
    """Factory function to create dataset adapters."""
    adapters = {
        "natural_questions": NaturalQuestionsAdapter,
        "stackoverflow": StackOverflowAdapter,
        "energy_papers": EnergyPapersAdapter,
    }
    
    if adapter_type not in adapters:
        available = ", ".join(adapters.keys())
        raise ValueError(f"Unknown adapter type '{adapter_type}'. Available: {available}")
    
    adapter_class = adapters[adapter_type]
    return adapter_class(dataset_path, version)


def cmd_ingest(args):
    """Run ingestion pipeline."""
    logger = logging.getLogger("ingest")
    logger.info(f"Starting ingestion: {args.adapter_type} from {args.dataset_path}")
    
    # Load configuration
    config = load_config(args.config) if args.config else load_config()
    
    # Create adapter
    adapter = get_adapter(args.adapter_type, args.dataset_path, args.version)
    
    # Create pipeline
    pipeline = IngestionPipeline(config=config)
    
    # Parse split
    split = DatasetSplit(args.split)
    
    # Run ingestion
    try:
        record = pipeline.ingest_dataset(
            adapter=adapter,
            split=split,
            dry_run=args.dry_run,
            max_documents=args.max_documents,
            canary_mode=args.canary
        )
        
        # Print results
        print(f"\n✓ Ingestion completed successfully!")
        print(f"  Dataset: {record.dataset_name} v{record.dataset_version}")
        print(f"  Documents: {record.total_documents}")
        print(f"  Chunks: {record.total_chunks}")
        print(f"  Successful: {record.successful_chunks}")
        print(f"  Failed: {record.failed_chunks}")
        print(f"  Success rate: {record.successful_chunks/record.total_chunks*100:.1f}%" if record.total_chunks > 0 else "N/A")
        print(f"  Run ID: {record.run_id}")
        
        if args.verify:
            logger.info("Running verification...")
            collection_info = pipeline.get_collection_status()
            print(f"\n📊 Collection Status:")
            print(f"  Name: {collection_info.get('collection_name', 'unknown')}")
            print(f"  Points: {collection_info.get('points_count', 0)}")
            print(f"  Status: {collection_info.get('status', 'unknown')}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        print(f"\n✗ Ingestion failed: {e}")
        return 1


def cmd_batch_ingest(args):
    """Run batch ingestion for multiple datasets."""
    logger = logging.getLogger("batch_ingest")
    
    # Load batch configuration
    with open(args.batch_config, 'r') as f:
        batch_config = json.load(f)
    
    datasets = batch_config.get("datasets", [])
    if not datasets:
        logger.error("No datasets specified in batch configuration")
        return 1
    
    logger.info(f"Starting batch ingestion of {len(datasets)} datasets")
    
    # Create adapters
    adapters = []
    for dataset_config in datasets:
        adapter = get_adapter(
            dataset_config["type"],
            dataset_config["path"],
            dataset_config.get("version", "1.0.0")
        )
        adapters.append(adapter)
    
    # Run batch ingestion
    pipeline = BatchIngestionPipeline(args.config)
    
    try:
        results = pipeline.ingest_multiple_datasets(
            adapters=adapters,
            split=DatasetSplit(args.split),
            dry_run=args.dry_run,
            max_documents=args.max_documents
        )
        
        # Print summary
        summary = pipeline.get_summary()
        print(f"\n✓ Batch ingestion completed!")
        print(f"  Datasets processed: {summary['total_datasets']}")
        print(f"  Total documents: {summary['total_documents']}")
        print(f"  Total chunks: {summary['total_chunks']}")
        print(f"  Overall success rate: {summary['success_rate']*100:.1f}%")
        
        return 0
        
    except Exception as e:
        logger.error(f"Batch ingestion failed: {e}")
        print(f"\n✗ Batch ingestion failed: {e}")
        return 1


def cmd_evaluate(args):
    """Run evaluation on ingested dataset."""
    logger = logging.getLogger("evaluate")
    logger.info(f"Starting evaluation: {args.adapter_type}")
    
    # Load configuration
    config = load_config(args.config) if args.config else load_config()
    
    # Create adapter
    adapter = get_adapter(args.adapter_type, args.dataset_path, args.version)
    
    # Create retriever
    from retrievers.router import RetrieverRouter
    retriever = RetrieverRouter(config)
    
    # Create evaluator
    evaluator = RetrievalEvaluator(config)
    
    try:
        # Run evaluation
        evaluation_run = evaluator.evaluate_dataset(
            adapter=adapter,
            retriever=retriever,
            split=args.split
        )
        
        # Save results
        output_dir = Path(args.output_dir)
        evaluator.save_results(evaluation_run, output_dir)
        
        # Print summary
        metrics = evaluation_run.metrics
        print(f"\n✓ Evaluation completed!")
        print(f"  Dataset: {evaluation_run.dataset_name}")
        print(f"  Queries: {metrics.total_queries}")
        print(f"  Recall@5: {metrics.recall_at_k.get(5, 0):.3f}")
        print(f"  Precision@5: {metrics.precision_at_k.get(5, 0):.3f}")
        print(f"  NDCG@5: {metrics.ndcg_at_k.get(5, 0):.3f}")
        print(f"  MRR: {metrics.mrr:.3f}")
        print(f"  Results saved to: {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"\n✗ Evaluation failed: {e}")
        return 1


def cmd_status(args):
    """Show collection and pipeline status."""
    config = load_config(args.config) if args.config else load_config()
    pipeline = IngestionPipeline(config=config)
    
    try:
        # Get collection info
        collection_info = pipeline.get_collection_status()
        
        print(f"\n📊 Collection Status:")
        print(f"  Name: {collection_info.get('collection_name', 'unknown')}")
        print(f"  Points: {collection_info.get('points_count', 0):,}")
        print(f"  Status: {collection_info.get('status', 'unknown')}")
        
        vectors_config = collection_info.get('vectors_config', {})
        sparse_vectors_config = collection_info.get('sparse_vectors_config', {})
        
        if vectors_config:
            print(f"  Dense vectors: {len(vectors_config)}")
            for name, config in vectors_config.items():
                print(f"    {name}: {config.size} dims")
        
        if sparse_vectors_config:
            print(f"  Sparse vectors: {len(sparse_vectors_config)}")
        
        # Show recent lineage files
        lineage_dir = Path("output/lineage")
        if lineage_dir.exists():
            lineage_files = sorted(lineage_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
            if lineage_files:
                print(f"\n📝 Recent Ingestion Runs:")
                for file_path in lineage_files[:5]:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        record = data.get("ingestion_record", {})
                        print(f"  {record.get('dataset_name', 'unknown')} - {record.get('started_at', 'unknown')}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Status check failed: {e}")
        return 1


def cmd_cleanup(args):
    """Clean up canary collections and temporary files."""
    config = load_config(args.config) if args.config else load_config()
    pipeline = IngestionPipeline(config=config)
    
    try:
        pipeline.cleanup_canary_collections()
        print("✓ Cleanup completed")
        return 0
    except Exception as e:
        print(f"✗ Cleanup failed: {e}")
        return 1


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Ingestion Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest Natural Questions dataset
  python bin/ingest.py ingest natural_questions /path/to/nq --config config.yml
  
  # Dry run with limited documents
  python bin/ingest.py ingest stackoverflow /path/to/so --dry-run --max-docs 100
  
  # Canary ingestion
  python bin/ingest.py ingest energy_papers papers/ --canary
  
  # Batch ingestion
  python bin/ingest.py batch-ingest batch_config.json
  
  # Evaluate retrieval
  python bin/ingest.py evaluate natural_questions /path/to/nq --output-dir results/
  
  # Check status
  python bin/ingest.py status
        """
    )
    
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a single dataset")
    ingest_parser.add_argument("adapter_type", choices=["natural_questions", "stackoverflow", "energy_papers"],
                              help="Dataset adapter type")
    ingest_parser.add_argument("dataset_path", help="Path to dataset")
    ingest_parser.add_argument("--version", default="1.0.0", help="Dataset version")
    ingest_parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all",
                              help="Dataset split to process")
    ingest_parser.add_argument("--dry-run", action="store_true", help="Don't upload to vector store")
    ingest_parser.add_argument("--max-docs", type=int, dest="max_documents", help="Maximum documents to process")
    ingest_parser.add_argument("--canary", action="store_true", help="Use canary collection")
    ingest_parser.add_argument("--verify", action="store_true", help="Run verification after ingestion")
    ingest_parser.set_defaults(func=cmd_ingest)
    
    # Batch ingest command
    batch_parser = subparsers.add_parser("batch-ingest", help="Ingest multiple datasets")
    batch_parser.add_argument("batch_config", help="JSON file with batch configuration")
    batch_parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    batch_parser.add_argument("--dry-run", action="store_true", help="Don't upload to vector store")
    batch_parser.add_argument("--max-docs", type=int, dest="max_documents", help="Maximum documents per dataset")
    batch_parser.set_defaults(func=cmd_batch_ingest)
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate retrieval performance")
    eval_parser.add_argument("adapter_type", choices=["natural_questions", "stackoverflow", "energy_papers"])
    eval_parser.add_argument("dataset_path", help="Path to dataset")
    eval_parser.add_argument("--version", default="1.0.0", help="Dataset version")
    eval_parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    eval_parser.add_argument("--output-dir", default="output/evaluation", help="Output directory for results")
    eval_parser.set_defaults(func=cmd_evaluate)
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show pipeline status")
    status_parser.set_defaults(func=cmd_status)
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up canary collections")
    cleanup_parser.set_defaults(func=cmd_cleanup)
    
    # Parse and execute
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Execute command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
