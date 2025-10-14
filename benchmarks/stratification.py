import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Dict, List
import ast


class StratifiedRAGDatasetSplitter:
    """
    Simple stratified train/test split for RAG hyperparameter optimization.

    Strategy:
    - 80% train (validation) set: for hyperparameter selection
    - 20% test set: for final validation
    - Stratification: question_type × primary_tag_category × answer_count_bin

    This simplified approach is suitable for hyperparameter optimization without
    model training, where a single stratified split provides sufficient statistical
    power for reliable configuration selection.
    """

    def __init__(self, dataset_path: str, random_state: int = 42):
        self.dataset_path = Path(dataset_path)
        self.random_state = random_state
        self.questions_df: pd.DataFrame = None
        self._answer_bins: pd.Series = None
        self._strat_key: pd.Series = None

    def load_dataset(self):
        """Load and preprocess the questions dataset."""
        questions_file = self.dataset_path / "question.csv"
        self.questions_df = pd.read_csv(questions_file)
        print(f"Loaded {len(self.questions_df)} questions")

        # Normalize ID
        if 'question_id' in self.questions_df.columns:
            self.questions_df['id'] = self.questions_df['question_id']
        elif 'id' not in self.questions_df.columns:
            raise ValueError("Missing 'question_id' or 'id' column.")

        # Ensure question_type exists
        if 'question_type' not in self.questions_df.columns:
            self.questions_df['question_type'] = 'General'

        # Tags → tags_parsed
        if 'tags_parsed' not in self.questions_df.columns:
            if 'tags' in self.questions_df.columns:
                self.questions_df['tags_parsed'] = self.questions_df['tags'].apply(
                    lambda x: [t.strip()
                               for t in str(x).split(';') if t.strip()]
                )
            else:
                self.questions_df['tags_parsed'] = [
                    [] for _ in range(len(self.questions_df))]

        # Primary tag category
        if 'primary_tag_category' not in self.questions_df.columns:
            self.questions_df['primary_tag_category'] = self.questions_df['tags_parsed'].apply(
                lambda x: x[0] if len(x) > 0 else 'Other'
            )

        # Compute answer_count from answer_posts if present
        if 'answer_posts' in self.questions_df.columns:
            def count_answers(x):
                if pd.isna(x) or not str(x).strip():
                    return 0
                if isinstance(x, list):
                    return len(x)
                try:
                    parsed = ast.literal_eval(x)
                    if isinstance(parsed, list):
                        return len(parsed)
                except Exception:
                    pass
                return 1
            self.questions_df['answer_count'] = self.questions_df['answer_posts'].apply(
                count_answers)
        else:
            self.questions_df['answer_count'] = 0

    def _create_answer_count_bins(self) -> pd.Series:
        """Categorize answer counts into bins."""
        def bin_count(c):
            if 1 <= c <= 3:
                return 'low'
            elif 4 <= c <= 6:
                return 'medium'
            elif c >= 7:
                return 'high'
            else:
                return 'none'

        bins = self.questions_df['answer_count'].apply(bin_count)
        return bins

    def _create_strat_key(self, top_k_categories: int = 6) -> pd.Series:
        """
        Create a multi-dimensional stratification key.

        The key combines three dimensions to ensure representativeness:
        - question_type: Type of question
        - primary_tag_category: Main topic category (grouped)
        - answer_count_bin: Answer count category
        """
        cats = self.questions_df['primary_tag_category'].value_counts().head(
            top_k_categories).index
        grouped = self.questions_df['primary_tag_category'].apply(
            lambda x: x if x in cats else 'Other'
        )
        bins = self._create_answer_count_bins()
        self._answer_bins = bins
        strat = (
            self.questions_df['question_type'].astype(str) + "_" +
            grouped.astype(str) + "_" +
            bins.astype(str)
        )
        return strat

    def create_train_test_split(
        self,
        test_size: float = 0.2,
        min_samples_per_stratum: int = 5
    ) -> Dict:
        """
        Create a stratified train/test split.

        Methodology:
        - 80% train (validation): for hyperparameter selection
        - 20% test: for final validation
        - Stratification preserves the distribution of characteristics

        Advantages:
        - Simplicity: Single split, no multiple folds
        - Efficiency: 4× faster than 5-fold CV
        - Sufficient statistical power: 80% = ~450 samples for validation
        - Stratification: Ensures representativeness

        Args:
            test_size: Fraction for test set (default: 0.2 = 20%)
            min_samples_per_stratum: Minimum samples per stratum (default: 5)

        Returns:
            Dictionary with train/test splits and statistics
        """
        if self.questions_df is None:
            self.load_dataset()

        strat_key = self._create_strat_key()
        self._strat_key = strat_key

        # Filter strata with sufficient size
        # For train/test split we need at least 2 samples per stratum
        # but we use a higher threshold for safety
        counts = strat_key.value_counts()
        valid = counts[counts >= min_samples_per_stratum].index
        mask = strat_key.isin(valid)
        filtered = self.questions_df[mask].copy()
        strat_f = strat_key[mask]
        bins_f = self._answer_bins[mask]

        print(f"\n{'=' * 70}")
        print("STRATIFIED TRAIN/TEST SPLIT")
        print(f"{'=' * 70}")
        print(f"Total strata identified: {len(counts)}")
        print(
            f"Valid strata (≥{min_samples_per_stratum} samples): {len(valid)}")
        print(f"Samples retained: {len(filtered)} / {len(self.questions_df)}")
        print(
            f"Split ratio: {int((1 - test_size) * 100)}/{int(test_size * 100)}")
        print(f"{'=' * 70}\n")

        id_col = 'question_id' if 'question_id' in filtered.columns else 'id'

        # Stratified train/test split
        train_df, test_df = train_test_split(
            filtered,
            test_size=test_size,
            stratify=strat_f,
            random_state=self.random_state
        )

        # Store indices for reference
        train_strat = strat_f.loc[train_df.index]
        test_strat = strat_f.loc[test_df.index]
        train_bins = bins_f.loc[train_df.index]
        test_bins = bins_f.loc[test_df.index]

        # Create splits dictionary
        splits = {
            'train': train_df[id_col].tolist(),
            'test': test_df[id_col].tolist()
        }

        # Statistics
        statistics = {
            'train': {
                'size': len(train_df),
                'percentage': len(train_df) / len(filtered) * 100,
                'question_types': train_df['question_type'].value_counts().to_dict(),
                'answer_bins': train_bins.value_counts().to_dict(),
                'strata_distribution': train_strat.value_counts().to_dict()
            },
            'test': {
                'size': len(test_df),
                'percentage': len(test_df) / len(filtered) * 100,
                'question_types': test_df['question_type'].value_counts().to_dict(),
                'answer_bins': test_bins.value_counts().to_dict(),
                'strata_distribution': test_strat.value_counts().to_dict()
            }
        }

        # Metadata
        metadata = {
            'total_samples': int(len(filtered)),
            'train_samples': int(len(train_df)),
            'test_samples': int(len(test_df)),
            'test_size': float(test_size),
            'random_state': self.random_state,
            'stratification_groups': int(len(valid)),
            'stratification_key': 'question_type × primary_tag_category × answer_count_bin',
            'answer_count_bins': ['none (0)', 'low (1-3)', 'medium (4-6)', 'high (7+)'],
            'split_method': 'stratified_train_test_split',
            'creation_timestamp': pd.Timestamp.now().isoformat()
        }

        # Print results
        print(f"Split Results:")
        print(
            f"  Train: {len(train_df)} samples ({len(train_df) / len(filtered) * 100:.1f}%)")
        print(
            f"  Test:  {len(test_df)} samples ({len(test_df) / len(filtered) * 100:.1f}%)")

        print(f"\nStratification Verification:")
        print(f"  Number of strata: {len(valid)}")
        print(f"  Train strata coverage: {len(train_strat.unique())}")
        print(f"  Test strata coverage: {len(test_strat.unique())}")

        # Verify that distributions are similar
        train_type_dist = train_df['question_type'].value_counts(
            normalize=True)
        test_type_dist = test_df['question_type'].value_counts(normalize=True)

        print(f"\nQuestion Type Distribution:")
        for qtype in train_type_dist.index:
            train_pct = train_type_dist.get(qtype, 0) * 100
            test_pct = test_type_dist.get(qtype, 0) * 100
            print(f"  {qtype}: Train={train_pct:.1f}%, Test={test_pct:.1f}%")

        print(f"\n{'=' * 70}\n")

        return {
            'splits': splits,
            'statistics': statistics,
            'metadata': metadata
        }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Simple stratified train/test split for RAG hyperparameter optimization.")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to dataset root (expects question.csv)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test set size as fraction (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random state for reproducibility (default: 42)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for split JSON (optional)")
    args = parser.parse_args()

    splitter = StratifiedRAGDatasetSplitter(
        args.dataset_path,
        random_state=args.random_state
    )
    splitter.load_dataset()
    split_info = splitter.create_train_test_split(test_size=args.test_size)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(split_info, f, ensure_ascii=False, indent=2)
        print(f"✓ Split saved to: {output_path}")

    # Display summary
    print("\nSummary:")
    print(f"  Train samples: {split_info['metadata']['train_samples']}")
    print(f"  Test samples:  {split_info['metadata']['test_samples']}")
    print(f"  Random state:  {split_info['metadata']['random_state']}")
