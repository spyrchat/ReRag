#!/usr/bin/env python3
"""
AutoRAG Multi-Arm Bandit Optimization Runner

This script provides a command-line interface for running hyperparameter optimization
using multi-arm bandit algorithms on your modular RAG system.
"""

import argparse
import sys
import yaml
from pathlib import Path

from .autorag_bandit_optimizer import (
    AutoRAGBanditOptimizer,
    UCBAlgorithm,
    ThompsonSamplingAlgorithm,
    EpsilonGreedyAlgorithm
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AutoRAG Multi-Arm Bandit Hyperparameter Optimization"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="Path to base configuration file"
    )

    parser.add_argument(
        "--search-space",
        type=str,
        default="benchmarks/hyperparameter_spaces.yml",
        help="Path to hyperparameter search space configuration"
    )

    parser.add_argument(
        "--space-name",
        type=str,
        default="grid_search_space",
        help="Name of search space to use from the spaces file"
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["ucb", "thompson", "epsilon_greedy"],
        default="ucb",
        help="Multi-arm bandit algorithm to use"
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum number of optimization iterations"
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience (iterations without improvement)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="autorag_optimization_results.yml",
        help="Output file for optimization results"
    )

    # Algorithm-specific parameters
    parser.add_argument(
        "--ucb-confidence",
        type=float,
        default=2.0,
        help="Confidence level for UCB algorithm"
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Epsilon value for epsilon-greedy algorithm"
    )

    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.01,
        help="Decay rate for epsilon in epsilon-greedy algorithm"
    )

    return parser.parse_args()


def load_search_space(search_space_path: str, space_name: str):
    """Load specific search space from configuration file."""
    with open(search_space_path, 'r') as f:
        # Load YAML with multiple documents
        spaces = list(yaml.safe_load_all(f))

    # Find the requested space
    for space in spaces:
        if space_name in space:
            return space[space_name]

    raise ValueError(
        f"Search space '{space_name}' not found in {search_space_path}")


def create_algorithm(args):
    """Create bandit algorithm based on arguments."""
    if args.algorithm == "ucb":
        return UCBAlgorithm(confidence_level=args.ucb_confidence)
    elif args.algorithm == "thompson":
        return ThompsonSamplingAlgorithm()
    elif args.algorithm == "epsilon_greedy":
        return EpsilonGreedyAlgorithm(
            epsilon=args.epsilon,
            decay_rate=args.epsilon_decay
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")


def main():
    """Main optimization runner."""
    args = parse_arguments()

    print("🤖 AutoRAG Multi-Arm Bandit Optimization")
    print("=" * 50)
    print(f"📁 Base config: {args.config}")
    print(f"🔍 Search space: {args.search_space} ({args.space_name})")
    print(f"🎯 Algorithm: {args.algorithm}")
    print(f"🔄 Max iterations: {args.max_iterations}")
    print("=" * 50)

    # Load search space configuration
    try:
        space_config = load_search_space(args.search_space, args.space_name)
        print(
            f"✅ Loaded search space: {space_config.get('description', 'No description')}")
    except Exception as e:
        print(f"❌ Error loading search space: {e}")
        sys.exit(1)

    # Create temporary search space file
    temp_space_file = "temp_search_space.yml"
    with open(temp_space_file, 'w') as f:
        yaml.dump(space_config, f)

    try:
        # Create algorithm
        algorithm = create_algorithm(args)
        print(f"🧠 Using algorithm: {algorithm.get_name()}")

        # Create optimizer
        optimizer = AutoRAGBanditOptimizer(
            fallback_config_path=args.config,
            hyperparameter_space_path=temp_space_file,
            algorithm=algorithm,
            max_iterations=args.max_iterations,
            early_stopping_patience=args.early_stopping_patience
        )

        # Run optimization
        print("\n🚀 Starting optimization...")
        result = optimizer.optimize()

        # Save results
        optimizer.save_results(result, args.output)

        print("\n✅ Optimization completed!")
        print(f"🏆 Best configuration: {result.best_arm.arm_id}")
        print(f"📊 Best performance: {result.best_arm.avg_reward:.4f}")
        print(f"🔄 Total iterations: {result.total_iterations}")
        print(f"⏱️  Total time: {result.total_evaluation_time:.2f}s")
        print(f"💾 Results saved to: {args.output}")

        # Display top configurations
        print("\n📈 Top 5 Configurations:")
        print("-" * 50)
        all_arms = optimizer.hyperparameter_space.arms
        top_arms = sorted(
            all_arms, key=lambda x: x.avg_reward, reverse=True)[:5]

        for i, arm in enumerate(top_arms, 1):
            if arm.num_pulls > 0:
                print(
                    f"{i}. {arm.arm_id}: {arm.avg_reward:.4f} (pulls: {arm.num_pulls})")

    except KeyboardInterrupt:
        print("\n⏹️  Optimization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        sys.exit(1)
    finally:
        # Clean up temporary file
        if Path(temp_space_file).exists():
            Path(temp_space_file).unlink()


if __name__ == "__main__":
    main()
