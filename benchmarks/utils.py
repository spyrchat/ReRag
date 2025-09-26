from typing import List


def get_chunk_ids_for_external_id(qdrant_client, collection_name: str, external_id: str) -> List[str]:
    """
    Retrieve all chunk_ids for a given external_id from a specified Qdrant collection.
    Args:
        qdrant_client: QdrantClient instance (already connected)
        collection_name: Name of the Qdrant collection to query
        external_id: The parent document/answer ID (e.g., 'a_2157446')
    Returns:
        List of chunk_id strings
    """
    filter = {
        "must": [
            {"key": "external_id", "match": {"value": external_id}}
        ]
    }
    chunk_ids = []
    scroll_result = qdrant_client.scroll(
        collection_name=collection_name,
        scroll_filter=filter,
        with_payload=True,
    )
    for point in scroll_result[0]:
        chunk_id = point.payload.get("chunk_id")
        if chunk_id:
            chunk_ids.append(chunk_id)
    return chunk_ids


def preload_chunk_id_mapping(qdrant_client, collection_name: str) -> dict:
    """
    Preload all chunk_ids and their external_ids from Qdrant collection.
    Returns: dict mapping external_id -> list of chunk_ids
    """
    mapping = {}
    # Scroll through all points in the collection
    next_page = None
    while True:
        scroll_result = qdrant_client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=1000,
            offset=next_page
        )
        points, next_page = scroll_result
        for point in points:
            external_id = point.payload.get("external_id")
            chunk_id = point.payload.get("chunk_id")
            if external_id and chunk_id:
                mapping.setdefault(external_id, []).append(chunk_id)
        if not next_page:
            break
    return mapping
