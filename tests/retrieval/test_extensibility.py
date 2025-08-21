#!/usr/bin/env python3
"""
Demonstration of retrieval pipeline extensibility and configuration-driven setup.
Shows how easy it is to create, modify, and extend retrieval pipelines.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import yaml
import logging
from pathlib import Path
from components.retrieval_pipeline import RetrievalPipelineFactory

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_configuration_driven_pipelines():
    """Test creating pipelines from YAML configurations."""
    print("=" * 60)
    print("CONFIGURATION-DRIVEN RETRIEVAL PIPELINES")
    print("=" * 60)
    
    config_dir = Path("pipelines/configs/retrieval")
    test_query = "How to handle Python exceptions properly?"
    
    configs = [
        ("basic_dense.yml", "🔍 Basic Dense Retrieval"),
        ("advanced_reranked.yml", "🎯 Advanced with Reranking"),
        ("experimental.yml", "🧪 Experimental Configuration")
    ]
    
    for config_file, description in configs:
        print(f"\n{description}")
        print("-" * 40)
        
        try:
            # Load configuration
            config_path = config_dir / config_file
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Create pipeline from config
            pipeline = RetrievalPipelineFactory.create_from_config(config)
            
            print(f"📋 Configuration: {config_file}")
            print(f"🔧 Components: {[c.component_name for c in pipeline.components]}")
            
            # Run retrieval
            results = pipeline.run(test_query, k=3)
            
            print(f"📊 Results: {len(results)}")
            for i, result in enumerate(results[:2], 1):  # Show top 2
                labels = result.document.metadata.get('labels', {})
                print(f"  {i}. Score: {result.score:.3f} | Method: {result.retrieval_method}")
                print(f"     Question: {labels.get('title', 'N/A')[:50]}...")
            
        except Exception as e:
            print(f"❌ Error with {config_file}: {e}")


def demonstrate_runtime_extensibility():
    """Show how easy it is to modify pipelines at runtime."""
    print("\n" + "=" * 60)
    print("RUNTIME PIPELINE MODIFICATION")
    print("=" * 60)
    
    # Load basic config
    with open("pipelines/configs/retrieval/basic_dense.yml", 'r') as f:
        config = yaml.safe_load(f)
    
    pipeline = RetrievalPipelineFactory.create_from_config(config)
    test_query = "Python list comprehension performance"
    
    print("\n1️⃣ Original Pipeline")
    print(f"Components: {[c.component_name for c in pipeline.components]}")
    
    # Add components dynamically
    print("\n2️⃣ Adding Score Filter (1 line)")
    from components.filters import ScoreFilter
    pipeline.add_component(ScoreFilter(min_score=0.5))
    print(f"Components: {[c.component_name for c in pipeline.components]}")
    
    print("\n3️⃣ Adding Cross-Encoder Reranker (1 line)")
    try:
        from components.rerankers import CrossEncoderReranker
        pipeline.add_component(CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_k=3
        ))
        print(f"Components: {[c.component_name for c in pipeline.components]}")
    except ImportError:
        print("❌ CrossEncoder not available")
    
    print("\n4️⃣ Adding Answer Enhancer (1 line)")
    from components.filters import AnswerEnhancer
    pipeline.add_component(AnswerEnhancer())
    print(f"Components: {[c.component_name for c in pipeline.components]}")
    
    # Test the enhanced pipeline
    print("\n5️⃣ Testing Enhanced Pipeline")
    try:
        results = pipeline.run(test_query, k=5)
        print(f"Results: {len(results)}")
        
        if results:
            result = results[0]
            print(f"Top result:")
            print(f"  Score: {result.score:.3f}")
            print(f"  Method: {result.retrieval_method}")
            print(f"  Enhanced: {result.metadata.get('enhanced', False)}")
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")


def demonstrate_component_swapping():
    """Show how to swap components easily."""
    print("\n" + "=" * 60)
    print("COMPONENT SWAPPING")
    print("=" * 60)
    
    # Load config
    with open("pipelines/configs/retrieval/basic_dense.yml", 'r') as f:
        config = yaml.safe_load(f)
    
    pipeline = RetrievalPipelineFactory.create_from_config(config)
    
    # Add a reranker
    from components.rerankers import CrossEncoderReranker
    cross_encoder = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    pipeline.add_component(cross_encoder)
    
    print("🔄 Original with CrossEncoder:")
    print(f"Components: {[c.component_name for c in pipeline.components]}")
    
    # Swap reranker
    print("\n🔄 Swapping to BGE Reranker:")
    try:
        # Remove old reranker
        pipeline.remove_component("cross_encoder_reranker_ms-marco-MiniLM-L-6-v2")
        
        # Add new reranker
        from components.advanced_rerankers import BgeReranker
        bge_reranker = BgeReranker(model_name="BAAI/bge-reranker-base")
        pipeline.add_component(bge_reranker)
        
        print(f"Components: {[c.component_name for c in pipeline.components]}")
        print("✅ Reranker swapped successfully!")
        
    except ImportError:
        print("❌ BGE reranker not available")
    except Exception as e:
        print(f"❌ Swapping failed: {e}")


def create_custom_component_example():
    """Show how to create and add a custom component."""
    print("\n" + "=" * 60)
    print("CUSTOM COMPONENT CREATION")
    print("=" * 60)
    
    from components.retrieval_pipeline import PostProcessor, RetrievalResult
    from typing import List
    
    class CustomScoreBooster(PostProcessor):
        """Custom component that boosts scores for Python-related content."""
        
        def __init__(self, boost_factor: float = 1.2):
            self.boost_factor = boost_factor
        
        @property
        def component_name(self) -> str:
            return f"python_score_booster_{self.boost_factor}"
        
        def post_process(self, query: str, results: List[RetrievalResult], **kwargs) -> List[RetrievalResult]:
            """Boost scores for Python-related answers."""
            for result in results:
                labels = result.document.metadata.get('labels', {})
                tags = labels.get('tags', [])
                
                # Boost score if Python-related
                if any('python' in tag.lower() for tag in tags):
                    result.score *= self.boost_factor
                    result.metadata['score_boosted'] = True
                    result.retrieval_method = f"{result.retrieval_method}+python_boost"
            
            # Re-sort by score
            results.sort(key=lambda x: x.score, reverse=True)
            return results
    
    print("📝 Created custom PostProcessor: CustomScoreBooster")
    print("💡 Boosts scores for Python-related content")
    
    # Test the custom component
    with open("pipelines/configs/retrieval/basic_dense.yml", 'r') as f:
        config = yaml.safe_load(f)
    
    pipeline = RetrievalPipelineFactory.create_from_config(config)
    
    # Add custom component
    custom_booster = CustomScoreBooster(boost_factor=1.5)
    pipeline.add_component(custom_booster)
    
    print(f"\n🔧 Pipeline with custom component:")
    print(f"Components: {[c.component_name for c in pipeline.components]}")
    print("\n✅ Custom component added in just 1 line!")
    print("💡 This shows how easy it is to extend the system")


def main():
    """Run all extensibility demonstrations."""
    print("🚀 RETRIEVAL PIPELINE EXTENSIBILITY DEMONSTRATION")
    print("Showing how easy it is to create, modify, and extend pipelines")
    
    try:
        test_configuration_driven_pipelines()
        demonstrate_runtime_extensibility()
        demonstrate_component_swapping()
        create_custom_component_example()
        
        print("\n" + "=" * 60)
        print("✅ EXTENSIBILITY SUMMARY")
        print("=" * 60)
        print("🔧 Configuration-driven: Change pipeline via YAML")
        print("⚡ Runtime modification: Add/remove components dynamically")
        print("🔄 Component swapping: Replace components easily")
        print("🎨 Custom components: Create new components in minutes")
        print("📁 Organized structure: Configs in pipelines/configs/retrieval/")
        print("🧪 Easy experimentation: Modify configs for A/B testing")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
