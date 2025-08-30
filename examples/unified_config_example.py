#!/usr/bin/env python3
"""
Example script demonstrating the unified configuration system.
Shows how to load configurations and create pipelines using the new approach.
"""

from components.retrieval_pipeline import RetrievalPipelineFactory
from config.config_loader import (
    load_config,
    get_retriever_config,
    get_benchmark_config,
    get_pipeline_config
)
import sys
import os
sys.path.append('/home/spiros/Desktop/Thesis/Thesis')


def main():
    """Demonstrate unified configuration usage."""

    print("🔧 Loading unified configuration...")

    # Load main configuration
    config = load_config("/home/spiros/Desktop/Thesis/Thesis/config.yml")
    print(f"✅ Loaded main config with sections: {list(config.keys())}")

    # Show available retrievers
    retrievers = config.get("retrievers", {})
    print(f"📋 Available retrievers: {list(retrievers.keys())}")

    # Test retriever config extraction
    print("\n🔍 Testing retriever configurations...")
    for retriever_type in ["dense", "sparse", "hybrid", "semantic"]:
        try:
            retriever_config = get_retriever_config(config, retriever_type)
            print(f"✅ {retriever_type}: {retriever_config.get('type')} retriever with {retriever_config.get('top_k', 'unknown')} top_k")

            # Show embedding config
            embedding = retriever_config.get("embedding", {})
            if isinstance(embedding, dict):
                if "provider" in embedding:
                    print(
                        f"   📡 Embedding: {embedding.get('provider')} - {embedding.get('model', 'N/A')}")
                elif "dense" in embedding:
                    dense = embedding.get("dense", {})
                    print(
                        f"   📡 Dense Embedding: {dense.get('provider')} - {dense.get('model', 'N/A')}")
                    sparse = embedding.get("sparse", {})
                    print(
                        f"   📡 Sparse Embedding: {sparse.get('provider')} - {sparse.get('model', 'N/A')}")

        except Exception as e:
            print(f"❌ {retriever_type}: {e}")

    # Test benchmark config
    print("\n📊 Testing benchmark configuration...")
    benchmark_config = get_benchmark_config(config)
    print(f"✅ Benchmark strategy: {benchmark_config['retrieval']['strategy']}")
    print(f"✅ Evaluation metrics: {benchmark_config['evaluation']['metrics']}")

    # Test pipeline config
    print("\n🏗️ Testing pipeline configuration...")
    pipeline_config = get_pipeline_config(config)
    print(f"✅ Default retriever: {pipeline_config['default_retriever']}")
    print(f"✅ Pipeline components: {len(pipeline_config['components'])}")

    # Test pipeline creation
    print("\n🚀 Testing pipeline creation...")
    try:
        # Create hybrid pipeline
        hybrid_pipeline = RetrievalPipelineFactory.create_from_unified_config(
            config, "hybrid")
        print(
            f"✅ Created hybrid pipeline with {len(hybrid_pipeline.components)} components")

        # Create dense pipeline
        dense_pipeline = RetrievalPipelineFactory.create_from_unified_config(
            config, "dense")
        print(
            f"✅ Created dense pipeline with {len(dense_pipeline.components)} components")

        # Create default pipeline (should use hybrid)
        default_pipeline = RetrievalPipelineFactory.create_from_unified_config(
            config)
        print(
            f"✅ Created default pipeline with {len(default_pipeline.components)} components")

    except Exception as e:
        print(f"❌ Pipeline creation failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n🎉 Unified configuration system test completed!")


if __name__ == "__main__":
    main()
