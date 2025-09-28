import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict, List, Any


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


def get_embedding_model(config: Dict[str, Any]) -> str:
    """Extract embedding model from config."""
    embedding = config.get('embedding', {})
    if 'dense' in embedding:
        return embedding['dense'].get('model', 'unknown')
    elif 'sparse' in embedding:
        return embedding['sparse'].get('model', 'unknown')
    return embedding.get('model', 'unknown')


def calculate_confidence_intervals(values: List[float], confidence: float = 0.95) -> Dict[str, float]:
    if not values or len(values) < 1:
        return {
            'mean': 0.0, 'std': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0,
            'median': 0.0, 'count': 0, 'margin_error': 0.0
        }

    # Special case for n=1
    if len(values) == 1:
        single_val = values[0] if not np.isnan(values[0]) else 0.0
        return {
            'mean': single_val, 'std': 0.0, 'ci_lower': single_val, 'ci_upper': single_val,
            'median': single_val, 'count': 1, 'margin_error': 0.0
        }
    values = [v for v in values if not np.isnan(v)]
    if not values:
        return {
            'mean': 0.0, 'std': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0,
            'median': 0.0, 'count': 0, 'margin_error': 0.0
        }
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0
    n = len(values)
    alpha = 1 - confidence
    degrees_freedom = n - 1
    t_critical = stats.t.ppf(1 - alpha/2, degrees_freedom) if n > 1 else 0
    margin_error = t_critical * (std / np.sqrt(n))
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error
    return {
        'mean': float(mean),
        'std': float(std),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'median': float(np.median(values)),
        'count': n,
        'margin_error': float(margin_error)
    }


def export_detailed_metrics(results: Dict[str, Any], filename: str, output_dir: Path):
    """Export detailed metrics with confidence intervals to CSV."""
    metrics_data = []
    for scenario_name, scenario_results in results.items():
        metrics = scenario_results.get('metrics', {})
        config = scenario_results.get('scenario_config', {})
        base_info = {
            'scenario': scenario_name,
            'retrieval_type': config.get('retrieval', {}).get('type', 'unknown'),
            'embedding_model': get_embedding_model(config),
            'total_queries': scenario_results.get('config', {}).get('total_queries', 0),
            'test_mode': scenario_results.get('test_mode', False),
            'timestamp': scenario_results.get('timestamp', ''),
        }
        for metric_name, metric_stats in metrics.items():
            if isinstance(metric_stats, dict) and 'mean' in metric_stats:
                row = base_info.copy()
                row.update({
                    'metric': metric_name,
                    'mean': metric_stats.get('mean', 0),
                    'std': metric_stats.get('std', 0),
                    'ci_lower': metric_stats.get('ci_lower', 0),
                    'ci_upper': metric_stats.get('ci_upper', 0),
                    'median': metric_stats.get('median', 0),
                    'min': metric_stats.get('min', 0),
                    'max': metric_stats.get('max', 0),
                    'count': metric_stats.get('count', 0),
                    'margin_error': metric_stats.get('margin_error', 0)
                })
                metrics_data.append(row)
    df = pd.DataFrame(metrics_data)
    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    print(f"💾 Detailed metrics exported to: {output_path}")
    return output_path


def export_per_query_results(results: Dict[str, Any], filename: str, output_dir: Path):
    """Export per-query metrics for all scenarios to a CSV file."""
    rows = []
    for scenario_name, scenario_results in results.items():
        per_query = scenario_results.get('per_query', [])
        config = scenario_results.get('scenario_config', {})
        for query_result in per_query:
            row = {
                'scenario': scenario_name,
                'retrieval_type': config.get('retrieval', {}).get('type', 'unknown'),
                'embedding_model': get_embedding_model(config),
                'query_id': query_result.get('query_id', ''),
                'query': query_result.get('query', ''),
                'relevant_docs': '|'.join(query_result.get('relevant_docs', [])),
                'retrieved_docs': '|'.join(query_result.get('retrieved_docs', [])),
            }
            metrics = query_result.get('metrics', {})
            for metric_name, value in metrics.items():
                row[metric_name] = value
            rows.append(row)
    df = pd.DataFrame(rows)
    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    print(f"💾 Per-query results exported to: {output_path}")
    return output_path
