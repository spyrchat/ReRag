#!/usr/bin/env python3
"""
Demonstration of the modular retrieval pipeline with easy component addition.
Shows how to build different pipeline configurations for various use cases.
"""

import yaml
from components.retrieval_pipeline import RetrievalPipelineFactory, RetrievalPipeline
from components.rerankers import CrossEncoderReranker, SemanticReranker, EnsembleReranker
from components.filters import ScoreFilter, TagFilter, DuplicateFilter, AnswerEnhancer, ResultLimiter

def test_modular_pipeline():
    print("=== Testing Modular Retrieval Pipeline ===")
    
    # Load config
    config_path = "pipelines/configs/stackoverflow_minilm.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Test 1: Basic Dense Pipeline
    print("\n1. Basic Dense Pipeline")
    print("-" * 40)
    
    dense_pipeline = RetrievalPipelineFactory.create_dense_pipeline(config)
    
    query = "How to count bits in Python?"
    results = dense_pipeline.run(query, k=3)
    
    print(f"Query: {query}")
    print(f"Results: {len(results)}")
    for i, result in enumerate(results, 1):
        labels = result.document.metadata.get('labels', {})
        print(f"  {i}. Score: {result.score:.3f} | Method: {result.retrieval_method}")
        print(f"     Question: {labels.get('title', 'N/A')}")
        print(f"     Tags: {labels.get('tags', [])}")
    
    # Test 2: Pipeline with Cross-Encoder Reranking
    print("\n2. Dense + Cross-Encoder Reranking")
    print("-" * 40)
    
    reranked_pipeline = RetrievalPipelineFactory.create_dense_pipeline(config)
    
    # Add cross-encoder reranker
    try:
        cross_encoder = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_k=3
        )
        reranked_pipeline.add_component(cross_encoder)
        
        results = reranked_pipeline.run(query, k=5)  # Retrieve 5, rerank to 3
        
        print(f"Query: {query}")
        print(f"Results: {len(results)}")
        for i, result in enumerate(results, 1):
            labels = result.document.metadata.get('labels', {})
            print(f"  {i}. Score: {result.score:.3f} | Method: {result.retrieval_method}")
            print(f"     Question: {labels.get('title', 'N/A')}")
            print(f"     Original Score: {result.metadata.get('original_score', 'N/A')}")
        
    except ImportError:
        print("❌ sentence-transformers not available for cross-encoder reranking")
    
    # Test 3: Multi-Component Pipeline
    print("\n3. Advanced Multi-Component Pipeline")
    print("-" * 40)
    
    advanced_pipeline = RetrievalPipelineFactory.create_dense_pipeline(config)
    
    # Add multiple components
    advanced_pipeline.add_component(ScoreFilter(min_score=0.3))
    advanced_pipeline.add_component(DuplicateFilter(dedup_by="external_id"))
    advanced_pipeline.add_component(TagFilter(excluded_tags=["deprecated"]))
    advanced_pipeline.add_component(AnswerEnhancer())
    advanced_pipeline.add_component(ResultLimiter(max_results=3))
    
    results = advanced_pipeline.run(query, k=10)
    
    print(f"Query: {query}")
    print(f"Results: {len(results)}")
    for i, result in enumerate(results, 1):
        labels = result.document.metadata.get('labels', {})
        print(f"  {i}. Score: {result.score:.3f} | Method: {result.retrieval_method}")
        print(f"     Question: {result.metadata.get('question_title', 'N/A')}")
        print(f"     Quality: {result.metadata.get('answer_quality', 'N/A')}")
        print(f"     Enhanced: {result.metadata.get('enhanced', False)}")
    
    # Test 4: Dynamic Pipeline Configuration
    print("\n4. Dynamic Pipeline Reconfiguration")
    print("-" * 40)
    
    dynamic_pipeline = RetrievalPipelineFactory.create_dense_pipeline(config)
    
    print("Initial components:", [c.component_name for c in dynamic_pipeline.components])
    
    # Add components dynamically
    dynamic_pipeline.add_component(ScoreFilter(min_score=0.5), position=1)  # Insert after retriever
    dynamic_pipeline.add_component(AnswerEnhancer())  # Append to end
    
    print("After adding components:", [c.component_name for c in dynamic_pipeline.components])
    
    # Remove a component
    dynamic_pipeline.remove_component("score_filter_0.5")
    print("After removing score filter:", [c.component_name for c in dynamic_pipeline.components])
    
    # Test 5: LangChain Compatibility
    print("\n5. LangChain Compatibility")
    print("-" * 40)
    
    langchain_retriever = dynamic_pipeline.to_langchain_retriever()
    langchain_results = langchain_retriever.get_relevant_documents(query)
    
    print(f"LangChain retriever returned {len(langchain_results)} documents")
    for i, doc in enumerate(langchain_results[:2], 1):
        labels = doc.metadata.get('labels', {})
        print(f"  {i}. Question: {labels.get('title', 'N/A')}")
        print(f"     Tags: {labels.get('tags', [])}")
    
    # Test 6: Configuration-based Pipeline
    print("\n6. Configuration-based Pipeline")
    print("-" * 40)
    
    # Define pipeline configuration
    pipeline_config = {
        "retriever": {"type": "dense", "k": 8},
        "score_filter_0.5": {"min_score": 0.5},
        "answer_enhancer": {},
        "result_limiter_5": {"max_results": 5}
    }
    
    config_pipeline = RetrievalPipelineFactory.create_dense_pipeline(config)
    config_pipeline.add_component(ScoreFilter())
    config_pipeline.add_component(AnswerEnhancer())
    config_pipeline.add_component(ResultLimiter())
    
    # Pass component-specific config
    results = config_pipeline.run(query, **pipeline_config)
    
    print(f"Configuration-based pipeline returned {len(results)} results")
    for i, result in enumerate(results, 1):
        print(f"  {i}. Score: {result.score:.3f} | Enhanced: {result.metadata.get('enhanced', False)}")


def demonstrate_easy_component_addition():
    """Show how easy it is to add new components."""
    print("\n" + "="*60)
    print("DEMONSTRATING EASY COMPONENT ADDITION")
    print("="*60)
    
    # Load base config
    config_path = "pipelines/configs/stackoverflow_minilm.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    query = "Python metaclasses explanation"
    
    # Scenario 1: Basic retrieval
    print("\n📊 Scenario 1: Basic Retrieval Only")
    basic_pipeline = RetrievalPipelineFactory.create_dense_pipeline(config)
    basic_results = basic_pipeline.run(query, k=3)
    print(f"   Components: {[c.component_name for c in basic_pipeline.components]}")
    print(f"   Results: {len(basic_results)}")
    
    # Scenario 2: Add score filtering (1 line of code!)
    print("\n🔍 Scenario 2: + Score Filtering")
    filtered_pipeline = RetrievalPipelineFactory.create_dense_pipeline(config)
    filtered_pipeline.add_component(ScoreFilter(min_score=0.6))  # ← Just 1 line!
    filtered_results = filtered_pipeline.run(query, k=10)
    print(f"   Components: {[c.component_name for c in filtered_pipeline.components]}")
    print(f"   Results: {len(filtered_results)} (filtered from 10)")
    
    # Scenario 3: Add reranking (1 more line!)
    print("\n🎯 Scenario 3: + Cross-Encoder Reranking")
    try:
        reranked_pipeline = RetrievalPipelineFactory.create_dense_pipeline(config)
        reranked_pipeline.add_component(ScoreFilter(min_score=0.4))
        reranked_pipeline.add_component(CrossEncoderReranker(top_k=3))  # ← Just 1 line!
        reranked_results = reranked_pipeline.run(query, k=10)
        print(f"   Components: {[c.component_name for c in reranked_pipeline.components]}")
        print(f"   Results: {len(reranked_results)} (reranked)")
        print(f"   Top result score: {reranked_results[0].score:.3f} (method: {reranked_results[0].retrieval_method})")
    except ImportError:
        print("   ❌ Cross-encoder not available")
    
    # Scenario 4: Add tag filtering and enhancement (2 more lines!)
    print("\n✨ Scenario 4: + Tag Filtering + Answer Enhancement")
    enhanced_pipeline = RetrievalPipelineFactory.create_dense_pipeline(config)
    enhanced_pipeline.add_component(ScoreFilter(min_score=0.4))
    enhanced_pipeline.add_component(TagFilter(required_tags=["python"]))  # ← Just 1 line!
    enhanced_pipeline.add_component(AnswerEnhancer())  # ← Just 1 line!
    enhanced_pipeline.add_component(ResultLimiter(max_results=3))
    enhanced_results = enhanced_pipeline.run(query, k=10)
    print(f"   Components: {[c.component_name for c in enhanced_pipeline.components]}")
    print(f"   Results: {len(enhanced_results)} (Python-only, enhanced)")
    if enhanced_results:
        print(f"   Top result quality: {enhanced_results[0].metadata.get('answer_quality', 'N/A')}")
    
    print(f"\n💡 Key Points:")
    print(f"   • Adding ANY component = just 1 line: pipeline.add_component(Component())")
    print(f"   • Components are reusable and configurable")
    print(f"   • Pipeline order matters and can be controlled")
    print(f"   • Easy to experiment with different combinations")
    print(f"   • LangChain compatible out of the box")


if __name__ == "__main__":
    test_modular_pipeline()
    demonstrate_easy_component_addition()
