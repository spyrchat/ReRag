"""
Qdrant Inspection Tool - Explore and query your vector database
"""
import argparse
import json
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, Range, MatchValue


def inspect_collections(client: QdrantClient):
    """List all collections and their stats."""
    print("=== QDRANT COLLECTIONS ===")
    collections = client.get_collections()

    if not collections.collections:
        print("No collections found.")
        return

    for collection in collections.collections:
        info = client.get_collection(collection.name)
        print(f"\nCollection: {collection.name}")
        print(f"   Status: {info.status}")
        print(
            f"   Vectors: {info.vectors_count if info.vectors_count else 'Computing...'}")

        # Vector configuration
        if hasattr(info.config.params, 'vectors'):
            if isinstance(info.config.params.vectors, dict):
                for name, config in info.config.params.vectors.items():
                    print(
                        f"   Vector '{name}': {config.size}D, distance={config.distance}")
            else:
                print(
                    f"   Vector: {info.config.params.vectors.size}D, distance={info.config.params.vectors.distance}")


def browse_data(client: QdrantClient, collection_name: str, limit: int = 10):
    """Browse data in a collection."""
    print(f"\n=== BROWSING: {collection_name} ===")

    try:
        # Get collection info
        info = client.get_collection(collection_name)
        print(f"Status: {info.status}")
        print(
            f"Vectors: {info.vectors_count if info.vectors_count else 'Computing...'}")

        # Get sample points
        points, next_page_offset = client.scroll(
            collection_name=collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False  # Don't load vectors for browsing
        )

        print(f"\nSample documents ({len(points)} shown):")
        for i, point in enumerate(points, 1):
            payload = point.payload
            print(f"\n{i}. ID: {point.id}")
            print(f"   Source: {payload.get('source', 'unknown')}")
            print(f"   External ID: {payload.get('external_id', 'unknown')}")
            print(f"   Split: {payload.get('split', 'unknown')}")
            print(
                f"   Chunk: {payload.get('chunk_index', 0)}/{payload.get('num_chunks', 1)}")
            print(f"   Model: {payload.get('embedding_model', 'unknown')}")
            print(f"   Text: {payload.get('text', '')[:200]}...")

            # Show labels if available
            labels = payload.get('labels', {})
            if labels and isinstance(labels, dict):
                interesting_labels = {k: v for k, v in labels.items()
                                      if k in ['post_type', 'tags', 'doc_type', 'title']}
                if interesting_labels:
                    print(f"   Labels: {interesting_labels}")

    except Exception as e:
        print(f"Error browsing collection: {e}")


def search_collection(client: QdrantClient, collection_name: str, query: str, limit: int = 5):
    """Search collection using vector similarity."""
    print(f"\n=== SEARCHING: {collection_name} ===")
    print(f"Query: '{query}'")

    try:
        # We need to embed the query first
        # For now, let's try a simple vector search
        # This is a simplified version - in production you'd embed the query properly

        # Try to get some results using vector search
        from qdrant_client.http.models import SearchRequest

        # First, let's try a simple scroll to see what's there
        points, _ = client.scroll(
            collection_name=collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )

        if not points:
            print("No data found in collection.")
            return

        # For now, let's do a text-based filter as fallback
        print(
            f"\n� Showing {len(points)} sample documents (vector search not implemented in inspector):")
        for i, point in enumerate(points, 1):
            payload = point.payload
            text = payload.get('text', '')

            # Simple text matching
            if query.lower() in text.lower():
                print(f"\n{i}. Match (ID: {point.id})")
                print(f"   External ID: {payload.get('external_id')}")
                print(f"   Source: {payload.get('source')}")
                print(f"   Text: {text[:300]}...")

        print(f"\n💡 Note: This is text-based search. For semantic search, use the retrieval pipeline.")

    except Exception as e:
        print(f"Error searching collection: {e}")


def filter_by_metadata(client: QdrantClient, collection_name: str, key: str, value: str, limit: int = 10):
    """Filter points by metadata."""
    print(f"\n=== FILTERING: {collection_name} ===")
    print(f"Filter: {key} = '{value}'")

    try:
        points, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False
        )

        print(f"\nFound {len(points)} results:")
        for i, point in enumerate(points, 1):
            payload = point.payload
            print(f"\n{i}. ID: {point.id}")
            print(f"   {key}: {payload.get(key)}")
            print(f"   Text: {payload.get('text', '')[:200]}...")

    except Exception as e:
        print(f"Error filtering collection: {e}")


def collection_stats(client: QdrantClient, collection_name: str):
    """Show detailed statistics for a collection."""
    print(f"\n=== STATISTICS: {collection_name} ===")

    try:
        # Get all points to compute stats
        all_points, _ = client.scroll(
            collection_name=collection_name,
            limit=10000,  # Adjust based on your collection size
            with_payload=True,
            with_vectors=False
        )

        if not all_points:
            print("No data found.")
            return

        # Compute statistics
        sources = {}
        splits = {}
        models = {}
        post_types = {}

        total_chars = 0
        chunk_counts = []

        for point in all_points:
            payload = point.payload

            # Count by source
            source = payload.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1

            # Count by split
            split = payload.get('split', 'unknown')
            splits[split] = splits.get(split, 0) + 1

            # Count by model
            model = payload.get('embedding_model', 'unknown')
            models[model] = models.get(model, 0) + 1

            # Count by post type (if available)
            labels = payload.get('labels', {})
            if isinstance(labels, dict):
                post_type = labels.get('post_type', 'unknown')
                post_types[post_type] = post_types.get(post_type, 0) + 1

            # Text statistics
            text_len = len(payload.get('text', ''))
            total_chars += text_len

            # Chunk info
            num_chunks = payload.get('num_chunks', 1)
            chunk_counts.append(num_chunks)

        # Print statistics
        print(f"📊 Total documents: {len(all_points)}")
        print(
            f"📊 Average text length: {total_chars / len(all_points):.1f} characters")
        print(
            f"📊 Average chunks per document: {sum(chunk_counts) / len(chunk_counts):.1f}")

        print(f"\nSources:")
        for source, count in sources.items():
            print(f"   {source}: {count}")

        print(f"\nSplits:")
        for split, count in splits.items():
            print(f"   {split}: {count}")

        print(f"\nModels:")
        for model, count in models.items():
            print(f"   {model}: {count}")

        if post_types and any(pt != 'unknown' for pt in post_types.keys()):
            print(f"\nPost Types:")
            for post_type, count in post_types.items():
                if post_type != 'unknown':
                    print(f"   {post_type}: {count}")

    except Exception as e:
        print(f"Error computing statistics: {e}")


def main():
    parser = argparse.ArgumentParser(description="Qdrant Database Inspector")
    parser.add_argument("--host", default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port")

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    # List collections
    subparsers.add_parser("list", help="List all collections")

    # Browse data
    browse_parser = subparsers.add_parser(
        "browse", help="Browse collection data")
    browse_parser.add_argument("collection", help="Collection name")
    browse_parser.add_argument(
        "--limit", type=int, default=10, help="Number of documents to show")

    # Search
    search_parser = subparsers.add_parser("search", help="Search collection")
    search_parser.add_argument("collection", help="Collection name")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results")

    # Filter
    filter_parser = subparsers.add_parser("filter", help="Filter by metadata")
    filter_parser.add_argument("collection", help="Collection name")
    filter_parser.add_argument("key", help="Metadata key")
    filter_parser.add_argument("value", help="Metadata value")
    filter_parser.add_argument(
        "--limit", type=int, default=10, help="Number of results")

    # Statistics
    stats_parser = subparsers.add_parser(
        "stats", help="Show collection statistics")
    stats_parser.add_argument("collection", help="Collection name")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Connect to Qdrant
    try:
        client = QdrantClient(host=args.host, port=args.port)

        if args.command == "list":
            inspect_collections(client)

        elif args.command == "browse":
            browse_data(client, args.collection, args.limit)

        elif args.command == "search":
            search_collection(client, args.collection, args.query, args.limit)

        elif args.command == "filter":
            filter_by_metadata(client, args.collection,
                               args.key, args.value, args.limit)

        elif args.command == "stats":
            collection_stats(client, args.collection)

    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        print("Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")


if __name__ == "__main__":
    main()
