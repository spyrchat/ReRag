#!/usr/bin/env python3
"""
Demonstration of advanced reranking components.
Shows how easily new rerankers can be added to the modular pipeline.
"""

import yaml
from components.retrieval_pipeline import RetrievalPipelineFactory
from components.rerankers import CrossEncoderReranker, EnsembleReranker
from components.advanced_rerankers import BgeReranker, MultiStageReranker
from components.filters import ScoreFilter, AnswerEnhancer

def test_advanced_reranking():
    print("=== Advanced Reranking Demonstration ===")
    
    # Load config
    config_path = "pipelines/configs/stackoverflow_minilm.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    query = "How to implement binary search in Python?"
    
    # Test 1: BGE Reranker (if transformers available)
    print("\n1. BGE Reranker")
    print("-" * 40)
    
    try:
        pipeline = RetrievalPipelineFactory.create_dense_pipeline(config)
        
        # Add BGE reranker in just 1 line!
        bge_reranker = BgeReranker(model_name="BAAI/bge-reranker-base", top_k=3)
        pipeline.add_component(bge_reranker)
        
        results = pipeline.run(query, k=5)
        
        print(f"Query: {query}")
        print(f"Results: {len(results)}")
        for i, result in enumerate(results, 1):
            labels = result.document.metadata.get('labels', {})
            print(f"  {i}. BGE Score: {result.score:.3f} | Method: {result.retrieval_method}")
            print(f"     Original Score: {result.metadata.get('original_score', 'N/A'):.3f}")
            print(f"     Question: {labels.get('title', 'N/A')[:50]}...")
        
    except ImportError as e:
        print(f"❌ BGE reranker requires transformers: {e}")
    except Exception as e:
        print(f"❌ BGE reranker failed: {e}")
    
    # Test 2: Multi-Stage Reranking
    print("\n2. Multi-Stage Reranking")
    print("-" * 40)
    
    try:
        pipeline = RetrievalPipelineFactory.create_dense_pipeline(config)
        
        # Create a multi-stage reranker in just 3 lines!
        stage1 = CrossEncoderReranker(model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2")
        stage2 = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        multistage = MultiStageReranker(stage1, stage2, stage1_k=8, stage2_k=3)
        
        pipeline.add_component(multistage)
        
        results = pipeline.run(query, k=15)  # Start with many candidates
        
        print(f"Query: {query}")
        print(f"Results: {len(results)} (15 -> 8 -> 3)")
        for i, result in enumerate(results, 1):
            labels = result.document.metadata.get('labels', {})
            print(f"  {i}. Final Score: {result.score:.3f} | Method: {result.retrieval_method}")
            print(f"     Multistage: {result.metadata.get('multistage_reranking', False)}")
            print(f"     Question: {labels.get('title', 'N/A')[:50]}...")
        
    except Exception as e:
        print(f"❌ Multi-stage reranking failed: {e}")
    
    # Test 3: Ensemble + Advanced Pipeline
    print("\n3. Ensemble + Filtering + Enhancement")
    print("-" * 40)
    
    try:
        pipeline = RetrievalPipelineFactory.create_dense_pipeline(config)
        
        # Build a sophisticated pipeline in just a few lines!
        cross_encoder = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # Create ensemble reranker
        ensemble = EnsembleReranker(
            rerankers=[cross_encoder],
            weights=[1.0],
            aggregation_method="weighted_average"
        )
        
        # Add components step by step
        pipeline.add_component(ScoreFilter(min_score=0.4))     # Filter low scores
        pipeline.add_component(ensemble)                       # Ensemble reranking
        pipeline.add_component(AnswerEnhancer())              # Enhance metadata
        
        results = pipeline.run(query, k=10)
        
        print(f"Query: {query}")
        print(f"Results: {len(results)}")
        print(f"Pipeline: {[c.component_name for c in pipeline.components]}")
        
        for i, result in enumerate(results, 1):
            labels = result.document.metadata.get('labels', {})
            print(f"  {i}. Score: {result.score:.3f} | Method: {result.retrieval_method}")
            print(f"     Enhanced: {result.metadata.get('enhanced', False)}")
            print(f"     Quality: {result.metadata.get('answer_quality', 'N/A')}")
            print(f"     Question: {labels.get('title', 'N/A')[:50]}...")
        
    except Exception as e:
        print(f"❌ Advanced pipeline failed: {e}")
    
    # Test 4: Easy Component Swapping
    print("\n4. Easy Component Swapping")
    print("-" * 40)
    
    try:
        # Start with one reranker
        pipeline = RetrievalPipelineFactory.create_dense_pipeline(config)
        
        cross_encoder = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        pipeline.add_component(cross_encoder)
        
        print("Original pipeline:", [c.component_name for c in pipeline.components])
        
        # Remove and replace with different reranker
        pipeline.remove_component("cross_encoder_reranker_ms-marco-MiniLM-L-6-v2")
        
        # Try BGE reranker instead
        try:
            bge_reranker = BgeReranker(model_name="BAAI/bge-reranker-base")
            pipeline.add_component(bge_reranker)
            print("Swapped to BGE:", [c.component_name for c in pipeline.components])
        except:
            print("BGE not available, keeping original")
        
    except Exception as e:
        print(f"❌ Component swapping failed: {e}")

def show_reranker_comparison():
    """Compare different rerankers on the same query."""
    print("\n" + "="*60)
    print("RERANKER COMPARISON")
    print("="*60)
    
    config_path = "pipelines/configs/stackoverflow_minilm.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    query = "Python list comprehension vs map performance"
    rerankers = []
    
    # Test different rerankers if available
    try:
        rerankers.append(("CrossEncoder-Tiny", CrossEncoderReranker("cross-encoder/ms-marco-TinyBERT-L-2-v2")))
    except:
        pass
    
    try:
        rerankers.append(("CrossEncoder-Mini", CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")))
    except:
        pass
    
    try:
        rerankers.append(("BGE-Base", BgeReranker("BAAI/bge-reranker-base")))
    except:
        pass
    
    if not rerankers:
        print("❌ No rerankers available for comparison")
        return
    
    for name, reranker in rerankers:
        try:
            print(f"\n📊 {name} Results:")
            print("-" * 30)
            
            pipeline = RetrievalPipelineFactory.create_dense_pipeline(config)
            pipeline.add_component(reranker)
            
            results = pipeline.run(query, k=3)
            
            for i, result in enumerate(results, 1):
                labels = result.document.metadata.get('labels', {})
                print(f"  {i}. Score: {result.score:.3f}")
                print(f"     Question: {labels.get('title', 'N/A')[:60]}...")
        
        except Exception as e:
            print(f"❌ {name} failed: {e}")

if __name__ == "__main__":
    test_advanced_reranking()
    show_reranker_comparison()
    
    print("\n" + "="*60)
    print("💡 KEY TAKEAWAYS")
    print("="*60)
    print("✅ Any reranker = just 1 line: pipeline.add_component(Reranker())")
    print("✅ Multi-stage reranking = 3 lines of code")
    print("✅ Easy to swap/compare different rerankers")
    print("✅ All rerankers preserve original scores in metadata")
    print("✅ Graceful fallback when models aren't available")
    print("✅ Pipeline shows exactly which components were used")
