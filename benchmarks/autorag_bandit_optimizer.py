"""
AutoRAG Multi-Arm Bandit Hyperparameter Optimizer

This module implements multi-arm bandit algorithms for hyperparameter optimization
in RAG systems, leveraging the existing modular architecture.
"""

import numpy as np
import pandas as pd
import yaml
import json
from typing import Dict, List, Any, Tuple, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
import time
import logging
from scipy import stats

from .benchmark_optimizer import BenchmarkOptimizer
from config.config_loader import load_config


@dataclass
class HyperparameterArm:
    """Represents a single hyperparameter configuration (arm)."""
    arm_id: str
    config: Dict[str, Any]
    total_reward: float = 0.0
    num_pulls: int = 0
    avg_reward: float = 0.0
    confidence_bound: float = 0.0
    last_updated: float = 0.0

    def update_reward(self, reward: float):
        """Update arm statistics with new reward."""
        self.total_reward += reward
        self.num_pulls += 1
        self.avg_reward = self.total_reward / self.num_pulls
        self.last_updated = time.time()


@dataclass
class OptimizationResult:
    """Results from a hyperparameter optimization run."""
    best_arm: HyperparameterArm
    total_iterations: int
    total_evaluation_time: float
    convergence_history: List[Dict[str, Any]]
    final_performance: Dict[str, float]


class BanditAlgorithm(ABC):
    """Abstract base class for multi-arm bandit algorithms."""

    @abstractmethod
    def select_arm(self, arms: List[HyperparameterArm], iteration: int) -> int:
        """Select which arm to pull next."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get algorithm name."""
        pass


class UCBAlgorithm(BanditAlgorithm):
    """Upper Confidence Bound algorithm for bandit optimization."""

    def __init__(self, confidence_level: float = 2.0):
        self.confidence_level = confidence_level

    def select_arm(self, arms: List[HyperparameterArm], iteration: int) -> int:
        """Select arm using UCB strategy."""
        if iteration < len(arms):
            # Pull each arm once initially
            return iteration

        # Calculate UCB for each arm
        ucb_values = []
        total_pulls = sum(arm.num_pulls for arm in arms)

        for arm in arms:
            if arm.num_pulls == 0:
                ucb_values.append(float('inf'))
            else:
                confidence_bound = self.confidence_level * np.sqrt(
                    np.log(total_pulls) / arm.num_pulls
                )
                ucb = arm.avg_reward + confidence_bound
                ucb_values.append(ucb)

        return int(np.argmax(ucb_values))

    def get_name(self) -> str:
        return f"UCB-{self.confidence_level}"


class ThompsonSamplingAlgorithm(BanditAlgorithm):
    """Thompson Sampling algorithm for bandit optimization."""

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def select_arm(self, arms: List[HyperparameterArm], iteration: int) -> int:
        """Select arm using Thompson Sampling strategy."""
        if iteration < len(arms):
            return iteration

        # Sample from posterior for each arm
        samples = []
        for arm in arms:
            if arm.num_pulls == 0:
                samples.append(np.random.beta(
                    self.prior_alpha, self.prior_beta))
            else:
                # Convert rewards to success/failure counts
                successes = arm.avg_reward * arm.num_pulls
                failures = arm.num_pulls - successes
                alpha = self.prior_alpha + successes
                beta = self.prior_beta + failures
                samples.append(np.random.beta(alpha, beta))

        return int(np.argmax(samples))

    def get_name(self) -> str:
        return "ThompsonSampling"


class EpsilonGreedyAlgorithm(BanditAlgorithm):
    """Epsilon-greedy algorithm for bandit optimization."""

    def __init__(self, epsilon: float = 0.1, decay_rate: float = 0.0):
        self.epsilon = epsilon
        self.decay_rate = decay_rate

    def select_arm(self, arms: List[HyperparameterArm], iteration: int) -> int:
        """Select arm using epsilon-greedy strategy."""
        if iteration < len(arms):
            return iteration

        # Apply decay to epsilon
        current_epsilon = self.epsilon * np.exp(-self.decay_rate * iteration)

        if np.random.random() < current_epsilon:
            # Explore: random selection
            return np.random.randint(len(arms))
        else:
            # Exploit: select best arm
            best_arms = [i for i, arm in enumerate(arms)
                         if arm.avg_reward == max(arm.avg_reward for arm in arms)]
            return np.random.choice(best_arms)

    def get_name(self) -> str:
        return f"EpsilonGreedy-{self.epsilon}"


class HyperparameterSpace:
    """Defines the hyperparameter search space for RAG components."""

    def __init__(self, space_config: Dict[str, Any]):
        self.space_config = space_config
        self.arms = self._generate_arms()

    def _generate_arms(self) -> List[HyperparameterArm]:
        """Generate all possible hyperparameter combinations."""
        arms = []

        # For grid search approach
        if self.space_config.get('search_type') == 'grid':
            arms = self._generate_grid_arms()
        # For random sampling approach
        elif self.space_config.get('search_type') == 'random':
            arms = self._generate_random_arms()
        # For predefined configurations
        else:
            arms = self._generate_predefined_arms()

        return arms

    def _generate_grid_arms(self) -> List[HyperparameterArm]:
        """Generate arms using grid search over parameter space."""
        arms = []
        parameters = self.space_config.get('parameters', {})

        # Generate all combinations using recursive approach
        def generate_combinations(param_names, param_values, current_config=None):
            if current_config is None:
                current_config = {}

            if not param_names:
                arm_id = self._generate_arm_id(current_config)
                arms.append(HyperparameterArm(
                    arm_id=arm_id,
                    config=current_config.copy()
                ))
                return

            param_name = param_names[0]
            remaining_params = param_names[1:]

            for value in param_values[param_name]:
                current_config[param_name] = value
                generate_combinations(
                    remaining_params, param_values, current_config)
                del current_config[param_name]

        param_names = list(parameters.keys())
        generate_combinations(param_names, parameters)

        return arms

    def _generate_random_arms(self) -> List[HyperparameterArm]:
        """Generate arms using random sampling."""
        arms = []
        n_arms = self.space_config.get('n_random_arms', 50)
        parameters = self.space_config.get('parameters', {})

        for i in range(n_arms):
            config = {}
            for param_name, param_values in parameters.items():
                if isinstance(param_values, list):
                    config[param_name] = np.random.choice(param_values)
                elif isinstance(param_values, dict):
                    if param_values.get('type') == 'uniform':
                        config[param_name] = np.random.uniform(
                            param_values['low'], param_values['high']
                        )
                    elif param_values.get('type') == 'log_uniform':
                        config[param_name] = np.exp(np.random.uniform(
                            np.log(param_values['low']),
                            np.log(param_values['high'])
                        ))

            arm_id = self._generate_arm_id(config)
            arms.append(HyperparameterArm(arm_id=arm_id, config=config))

        return arms

    def _generate_predefined_arms(self) -> List[HyperparameterArm]:
        """Generate arms from predefined configurations."""
        arms = []
        configs = self.space_config.get('predefined_configs', [])

        for i, config in enumerate(configs):
            arm_id = config.get('name', f'config_{i}')
            arms.append(HyperparameterArm(
                arm_id=arm_id,
                config=config.get('parameters', {})
            ))

        return arms

    def _generate_arm_id(self, config: Dict[str, Any]) -> str:
        """Generate unique arm ID from configuration."""
        # Create a deterministic hash of the configuration
        config_str = json.dumps(config, sort_keys=True)
        return f"arm_{hash(config_str) % 1000000:06d}"


class AutoRAGBanditOptimizer:
    """Main class for AutoRAG hyperparameter optimization using multi-arm bandits."""

    def __init__(
        self,
        fallback_config_path: str = "config.yml",
        hyperparameter_space_path: str = None,
        algorithm: BanditAlgorithm = None,
        max_iterations: int = 100,
        early_stopping_patience: int = 10,
        early_stopping_threshold: float = 0.001
    ):
        # Store fallback config path but don't load it (only used when no scenario provided)
        self.fallback_config_path = fallback_config_path
        self.benchmark_optimizer = BenchmarkOptimizer(fallback_config_path)
        self.algorithm = algorithm or UCBAlgorithm()
        self.max_iterations = max_iterations
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold

        self.optimization_history = []
        self.logger = self._setup_logger()

        # Load hyperparameter space (logger is now available)
        if hyperparameter_space_path:
            with open(hyperparameter_space_path, 'r') as f:
                space_config = yaml.safe_load(f)
            self.hyperparameter_space = HyperparameterSpace(space_config)
        else:
            # Use default space with self-contained scenarios
            self.hyperparameter_space = self._create_default_space()

    def _create_default_space(self) -> HyperparameterSpace:
        """Create a default hyperparameter space based on self-contained scenarios."""
        default_space = {
            'search_type': 'predefined',
            'predefined_configs': []
        }

        # Load existing benchmark scenarios as predefined configurations
        # Each scenario is completely self-contained (no merging)
        scenario_dir = Path("benchmark_scenarios")
        if scenario_dir.exists():
            for scenario_file in scenario_dir.glob("*.yml"):
                try:
                    # Load and validate each scenario is self-contained
                    scenario_config = self.benchmark_optimizer.load_benchmark_config(str(scenario_file))
                    
                    default_space['predefined_configs'].append({
                        'name': scenario_file.stem,
                        'parameters': scenario_config
                    })
                    self.logger.info(f"✅ Loaded self-contained scenario: {scenario_file.stem}")
                except Exception as e:
                    self.logger.warning(f"❌ Skipping invalid scenario {scenario_file.stem}: {e}")

        if not default_space['predefined_configs']:
            self.logger.warning("⚠️ No valid self-contained scenarios found. Using fallback config.")
            # Only use fallback if no scenarios are available
            fallback_config = load_config(self.fallback_config_path)
            default_space['predefined_configs'].append({
                'name': 'fallback',
                'parameters': fallback_config
            })

        return HyperparameterSpace(default_space)

    def _setup_logger(self) -> logging.Logger:
        """Setup logging for optimization process."""
        logger = logging.getLogger('AutoRAGBanditOptimizer')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def optimize(self) -> OptimizationResult:
        """Run the multi-arm bandit optimization process."""
        self.logger.info(
            f"Starting AutoRAG optimization with {self.algorithm.get_name()}")
        self.logger.info(
            f"Search space: {len(self.hyperparameter_space.arms)} arms")

        arms = self.hyperparameter_space.arms
        start_time = time.time()
        best_reward = -float('inf')
        patience_counter = 0

        for iteration in range(self.max_iterations):
            # Select arm using bandit algorithm
            selected_arm_idx = self.algorithm.select_arm(arms, iteration)
            selected_arm = arms[selected_arm_idx]

            self.logger.info(
                f"Iteration {iteration + 1}: Selected arm {selected_arm.arm_id}")

            # Evaluate the selected configuration
            reward = self._evaluate_configuration(selected_arm.config)

            # Update arm statistics
            selected_arm.update_reward(reward)

            # Track optimization history
            current_best = max(arms, key=lambda x: x.avg_reward)
            self.optimization_history.append({
                'iteration': iteration + 1,
                'selected_arm': selected_arm.arm_id,
                'reward': reward,
                'best_arm': current_best.arm_id,
                'best_reward': current_best.avg_reward,
                'exploration_rate': self._calculate_exploration_rate(arms, iteration)
            })

            # Early stopping check
            if current_best.avg_reward > best_reward + self.early_stopping_threshold:
                best_reward = current_best.avg_reward
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                self.logger.info(
                    f"Early stopping at iteration {iteration + 1}")
                break

            self.logger.info(
                f"Current best: {current_best.arm_id} (reward: {current_best.avg_reward:.4f})")

        total_time = time.time() - start_time
        best_arm = max(arms, key=lambda x: x.avg_reward)

        # Final evaluation of best configuration
        final_performance = self._evaluate_configuration(
            best_arm.config, detailed=True)

        result = OptimizationResult(
            best_arm=best_arm,
            total_iterations=len(self.optimization_history),
            total_evaluation_time=total_time,
            convergence_history=self.optimization_history,
            final_performance=final_performance
        )

        self.logger.info(
            f"Optimization completed! Best arm: {best_arm.arm_id}")
        self.logger.info(f"Best performance: {final_performance}")

        return result

    def _evaluate_configuration(self, config: Dict[str, Any], detailed: bool = False) -> Union[float, Dict[str, float]]:
        """Evaluate a standalone configuration (no merging - config must be self-contained)."""
        try:
            # No merging! Config must be completely self-contained
            # This enforces the isolated configuration principle
            
            # Validate that config is self-contained
            self.benchmark_optimizer._validate_complete_config(config, "bandit_arm_config")
            
            # Run benchmark with this standalone configuration
            results = self.benchmark_optimizer.run_optimization_scenario(
                scenario_name="bandit_evaluation",
                config=config
            )

            # Extract reward (we'll use F1 score as primary metric)
            primary_metric = results.get('metrics', {}).get('f1', {})
            if isinstance(primary_metric, dict):
                # Use F1@10 as default reward
                reward = primary_metric.get('10', 0.0)
            else:
                reward = primary_metric

            if detailed:
                return results.get('metrics', {})
            else:
                return float(reward)

        except Exception as e:
            self.logger.error(f"Error evaluating self-contained configuration: {e}")
            return 0.0 if not detailed else {}

    def _calculate_exploration_rate(self, arms: List[HyperparameterArm], iteration: int) -> float:
        """Calculate current exploration rate."""
        if iteration == 0:
            return 1.0

        unique_arms_pulled = len([arm for arm in arms if arm.num_pulls > 0])
        return unique_arms_pulled / len(arms)

    def save_results(self, result: OptimizationResult, output_path: str):
        """Save optimization results to file."""
        results_data = {
            'algorithm': self.algorithm.get_name(),
            'best_configuration': asdict(result.best_arm),
            'optimization_summary': {
                'total_iterations': result.total_iterations,
                'total_evaluation_time': result.total_evaluation_time,
                'final_performance': result.final_performance
            },
            'convergence_history': result.convergence_history
        }

        with open(output_path, 'w') as f:
            yaml.dump(results_data, f, default_flow_style=False)

        # Also save as CSV for analysis
        df = pd.DataFrame(result.convergence_history)
        csv_path = output_path.replace('.yml', '_history.csv')
        df.to_csv(csv_path, index=False)

        self.logger.info(f"Results saved to {output_path} and {csv_path}")


def main():
    """Example usage of AutoRAG bandit optimizer."""
    # Example hyperparameter space configuration
    space_config = {
        'search_type': 'grid',
        'parameters': {
            'retrieval.top_k': [10, 20, 50],
            'retrieval.search_params.score_threshold': [0.0, 0.1, 0.2],
            'reranker.enabled': [True, False],
            'reranker.top_k': [5, 10, 15]
        }
    }

    # Save example space configuration
    with open('example_hyperparameter_space.yml', 'w') as f:
        yaml.dump(space_config, f)

    # Run optimization with different algorithms
    algorithms = [
        UCBAlgorithm(confidence_level=2.0),
        ThompsonSamplingAlgorithm(),
        EpsilonGreedyAlgorithm(epsilon=0.1, decay_rate=0.01)
    ]

    for algorithm in algorithms:
        optimizer = AutoRAGBanditOptimizer(
            hyperparameter_space_path='example_hyperparameter_space.yml',
            algorithm=algorithm,
            max_iterations=50
        )

        result = optimizer.optimize()

        output_path = f'autorag_optimization_{algorithm.get_name().lower()}.yml'
        optimizer.save_results(result, output_path)


if __name__ == "__main__":
    main()
