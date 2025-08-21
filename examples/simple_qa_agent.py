#!/usr/bin/env python3
"""
Example: Simple Agent Using Configurable Retrieval Pipeline
Shows how easy it is for an agent to use any YAML configuration.
"""

from bin.agent_retriever import ConfigurableRetrieverAgent, get_agent_with_config
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class SimpleQAAgent:
    """
    Simple Q&A agent that can use any retrieval configuration.
    Demonstrates how to build agents on top of the configurable retrieval system.
    """

    def __init__(self, config_name: str = "basic_dense"):
        """
        Initialize agent with a specific retrieval configuration.

        Args:
            config_name (str): Name of the retrieval configuration to use
        """
        self.retriever = get_agent_with_config(config_name)
        self.config_info = self.retriever.get_config_info()

        print(f"Agent initialized with {config_name} configuration")
        print(f"   Retriever: {self.config_info['retriever_type']}")
        print(f"   Stages: {self.config_info['num_stages']}")

    def answer_question(self, question: str, top_k: int = 3) -> str:
        """
        Answer a question using the configured retrieval pipeline.

        Args:
            question (str): User's question
            top_k (int): Number of documents to retrieve

        Returns:
            str: Formatted answer with sources
        """
        print(f"\nQuestion: {question}")

        # Retrieve relevant documents
        documents = self.retriever.retrieve(question, top_k=top_k)

        if not documents:
            return "No relevant documents found."

        # Format answer
        answer = f"Found {len(documents)} relevant sources:\n\n"

        for i, doc in enumerate(documents, 1):
            answer += f"{i}. **Score: {doc['score']:.3f}** | Method: {doc['retrieval_method']}\n"

            if doc['question_title']:
                answer += f"   📝 Question: {doc['question_title']}\n"

            if doc['tags']:
                answer += f"   🏷️  Tags: {', '.join(doc['tags'][:3])}\n"

            content_preview = doc['content'][:200] + \
                "..." if len(doc['content']) > 200 else doc['content']
            answer += f"   💡 Answer: {content_preview}\n\n"

        return answer

    def switch_configuration(self, config_name: str):
        """Switch to a different retrieval configuration."""
        config_path = f"pipelines/configs/retrieval/{config_name}.yml"
        self.retriever.switch_config(config_path)
        self.config_info = self.retriever.get_config_info()

        print(f"🔄 Switched to {config_name} configuration")
        print(f"   Retriever: {self.config_info['retriever_type']}")
        print(f"   Stages: {self.config_info['num_stages']}")


def main():
    """Demonstrate the Q&A agent with different configurations."""
    print("🚀 Simple Q&A Agent Demo")
    print("=" * 50)

    # Test questions
    questions = [
        "How to count bits in Python?",
        "What are Python metaclasses?",
        "How to handle exceptions in Python?"
    ]

    # Test different configurations
    configs = ["basic_dense", "advanced_reranked"]

    for config in configs:
        print(f"\n{'🔧 ' + config.upper() + ' CONFIGURATION'}")
        print("=" * 50)

        try:
            # Create agent with specific config
            agent = SimpleQAAgent(config)

            # Answer a question
            question = questions[0]
            answer = agent.answer_question(question, top_k=2)
            print(answer)

        except Exception as e:
            print(f"❌ Error with {config}: {e}")

    # Demonstrate configuration switching
    print("\n🔄 CONFIGURATION SWITCHING DEMO")
    print("=" * 50)

    try:
        agent = SimpleQAAgent("basic_dense")

        question = "Python list comprehension performance"

        # Answer with basic config
        print("\n1️⃣ With Basic Dense Configuration:")
        answer1 = agent.answer_question(question, top_k=1)
        print(answer1)

        # Switch to advanced config
        agent.switch_configuration("advanced_reranked")

        # Answer same question with advanced config
        print("\n2️⃣ With Advanced Reranked Configuration:")
        answer2 = agent.answer_question(question, top_k=1)
        print(answer2)

    except Exception as e:
        print(f"❌ Error in switching demo: {e}")

    print("\n💡 Key Benefits:")
    print("✅ Agent can use ANY retrieval configuration")
    print("✅ Switch configurations without restarting")
    print("✅ Easy to experiment with different pipelines")
    print("✅ Configuration changes via YAML files")
    print("✅ No code changes needed for new pipelines")


if __name__ == "__main__":
    main()
