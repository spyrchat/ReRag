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
from pipelines.adapters.stackoverflow import StackOverflowAdapter
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
        "stackoverflow": StackOverflowAdapter
    }

    if adapter_type not in adapters:
        available = ", ".join(adapters.keys())
        raise ValueError(
            f"Unknown adapter type '{adapter_type}'. Available: {available}")

    adapter_class = adapters[adapter_type]
    return adapter_class(dataset_path, version)


def cmd_ingest(args):
    """Run ingestion pipeline."""
    logger = logging.getLogger("ingest")
    logger.info(
        f"Starting ingestion: {args.adapter_type} from {args.dataset_path}")

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
        print(
            f"  Success rate: {record.successful_chunks / record.total_chunks * 100:.1f}%" if record.total_chunks > 0 else "N/A")
        print(f"  Run ID: {record.run_id}")

        if args.verify:
            logger.info("Running verification...")
            collection_info = pipeline.get_collection_status()
            print(f"\n Collection Status:")
            print(
                f"  Name: {collection_info.get('collection_name', 'unknown')}")
            print(f"  Points: {collection_info.get('points_count', 0)}")
            print(f"  Status: {collection_info.get('status', 'unknown')}")

        return 0

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        print(f"\n✗ Ingestion failed: {e}")
        return 1


def cmd_status(args):
    """Show collection and pipeline status."""
    config = load_config(args.config) if args.config else load_config()
    pipeline = IngestionPipeline(config=config)

    try:
        # Get collection info
        collection_info = pipeline.get_collection_status()

        print(f"\nCollection Status:")
        print(f"  Name: {collection_info.get('collection_name', 'unknown')}")
        print(f"  Points: {collection_info.get('points_count', 0):,}")
        print(f"  Status: {collection_info.get('status', 'unknown')}")

        vectors_config = collection_info.get('vectors_config', {})
        sparse_vectors_config = collection_info.get(
            'sparse_vectors_config', {})

        if vectors_config:
            print(f"  Dense vectors: {len(vectors_config)}")
            for name, config in vectors_config.items():
                print(f"    {name}: {config.size} dims")

        if sparse_vectors_config:
            print(f"  Sparse vectors: {len(sparse_vectors_config)}")

        # Show recent lineage files
        lineage_dir = Path("output/lineage")
        if lineage_dir.exists():
            lineage_files = sorted(lineage_dir.glob(
                "*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
            if lineage_files:
                print(f"\n📝 Recent Ingestion Runs:")
                for file_path in lineage_files[:5]:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        record = data.get("ingestion_record", {})
                        print(
                            f"  {record.get('dataset_name', 'unknown')} - {record.get('started_at', 'unknown')}")

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
  
  # Check status
  python bin/ingest.py status
        """
    )

    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--verbose", "-v",
                        action="store_true", help="Verbose logging")

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest a single dataset")
    ingest_parser.add_argument("adapter_type", choices=["natural_questions", "stackoverflow", "energy_papers"],
                               help="Dataset adapter type")
    ingest_parser.add_argument("dataset_path", help="Path to dataset")
    ingest_parser.add_argument(
        "--version", default="1.0.0", help="Dataset version")
    ingest_parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all",
                               help="Dataset split to process")
    ingest_parser.add_argument(
        "--dry-run", action="store_true", help="Don't upload to vector store")
    ingest_parser.add_argument(
        "--max-docs", type=int, dest="max_documents", help="Maximum documents to process")
    ingest_parser.add_argument(
        "--canary", action="store_true", help="Use canary collection")
    ingest_parser.add_argument(
        "--verify", action="store_true", help="Run verification after ingestion")
    ingest_parser.set_defaults(func=cmd_ingest)

    # Status command
    status_parser = subparsers.add_parser(
        "status", help="Show pipeline status")
    status_parser.set_defaults(func=cmd_status)

    # Cleanup command
    cleanup_parser = subparsers.add_parser(
        "cleanup", help="Clean up canary collections")
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
