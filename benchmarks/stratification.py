import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import json
import yaml
from typing import Dict, List, Tuple, Optional


class StratifiedRAGDatasetSplitter:
    """
    Creates stratified splits for RAG evaluation ensuring:
    - Balanced question types across folds
    - Balanced technology categories  
    - Balanced answer count bins (low/medium/high)
    - Reproducible splits with seeds
    """

    def __init__(self, dataset_path: str, random_state: int = 42):
        self.dataset_path = Path(dataset_path)
        self.random_state = random_state
        self.questions_df = None

    def load_dataset(self):
        """Load SOSum dataset."""
        questions_file = self.dataset_path / "questions.jsonl"

        questions = []
        with open(questions_file, 'r') as f:
            for line in f:
                questions.append(json.loads(line))

        self.questions_df = pd.DataFrame(questions)
        print(f"📊 Loaded {len(self.questions_df)} questions")

    def create_answer_count_bins(self) -> pd.Series:
        """
        Create answer count bins:
        - Low: 1-3 answers
        - Medium: 4-6 answers  
        - High: 7+ answers
        """
        def categorize_answer_count(count):
            if 1 <= count <= 3:
                return 'low'
            elif 4 <= count <= 6:
                return 'medium'
            else:  # 7+
                return 'high'

        answer_bins = self.questions_df['answer_count'].apply(
            categorize_answer_count)

        # Print distribution
        bin_counts = answer_bins.value_counts()
        print(f"📈 Answer Count Distribution:")
        print(f"  Low (1-3): {bin_counts.get('low', 0)} questions")
        print(f"  Medium (4-6): {bin_counts.get('medium', 0)} questions")
        print(f"  High (7+): {bin_counts.get('high', 0)} questions")

        return answer_bins

    def create_stratification_key(self) -> pd.Series:
        """
        Create composite stratification key based on:
        - Question type (Conceptual/How-to/Debug)
        - Primary technology category (top 6 categories)
        - Answer count bins (low/medium/high)
        """
        # Ensure question type exists
        if 'question_type_label' not in self.questions_df.columns:
            # Create basic type from tags/content
            self.questions_df['question_type_label'] = 'General'

        # Ensure primary technology category exists
        if 'primary_tag_category' not in self.questions_df.columns:
            self.questions_df['primary_tag_category'] = self.questions_df['tags_parsed'].apply(
                lambda x: x[0] if len(x) > 0 else 'Other'
            )

        # Group rare categories into 'Other' to ensure enough samples
        top_categories = self.questions_df['primary_tag_category'].value_counts().head(
            6).index
        self.questions_df['primary_tag_category_grouped'] = self.questions_df['primary_tag_category'].apply(
            lambda x: x if x in top_categories else 'Other'
        )

        # Create answer count bins
        answer_bins = self.create_answer_count_bins()

        # Create composite stratification key
        strat_key = (
            self.questions_df['question_type_label'].astype(str) + "_" +
            self.questions_df['primary_tag_category_grouped'].astype(str) + "_" +
            answer_bins.astype(str)
        )

        print(
            f"🔑 Created {len(strat_key.unique())} unique stratification groups")
        return strat_key

    def create_cv_splits(self, n_folds: int = 5, min_samples_per_fold: int = 8) -> Dict:
        """
        Create stratified K-fold splits optimized for hyperparameter optimization.

        Strategy:
        - 4 folds for optimization (train on 3, dev on 1)
        - 1 fold held out for final testing

        Args:
            n_folds: Number of CV folds (default: 5)
            min_samples_per_fold: Minimum samples per group per fold

        Returns:
            Dictionary with fold information and splits
        """
        if self.questions_df is None:
            self.load_dataset()

        # Create stratification key
        strat_key = self.create_stratification_key()

        # Filter groups to ensure enough samples for stratification
        strat_counts = strat_key.value_counts()
        valid_strats = strat_counts[strat_counts >=
                                    n_folds * min_samples_per_fold].index

        mask = strat_key.isin(valid_strats)
        filtered_df = self.questions_df[mask].copy()
        filtered_strat_key = strat_key[mask]

        print(f"📈 Stratification groups: {len(strat_counts)}")
        print(
            f"✅ Valid groups (≥{n_folds * min_samples_per_fold} samples): {len(valid_strats)}")
        print(f"📊 Samples after filtering: {len(filtered_df)}")

        # Create stratified folds
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=self.random_state)

        splits = {}
        fold_stats = []

        for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(filtered_df, filtered_strat_key)):
            fold_train_val = filtered_df.iloc[train_val_idx]
            fold_test = filtered_df.iloc[test_idx]

            if fold_idx == n_folds - 1:
                # Last fold: held out for final testing only
                splits[f'fold_{fold_idx}'] = {
                    'train': [],  # No training on final test fold
                    'dev': [],    # No dev on final test fold
                    'test': fold_test['id'].tolist(),
                    'role': 'final_test'
                }

                fold_stats.append({
                    'fold': fold_idx,
                    'role': 'final_test',
                    'train_size': 0,
                    'dev_size': 0,
                    'test_size': len(fold_test),
                    'test_question_types': fold_test['question_type_label'].value_counts().to_dict(),
                    'test_answer_bins': self.create_answer_count_bins().loc[fold_test.index].value_counts().to_dict()
                })
            else:
                # Optimization folds: split train_val into train (75%) and dev (25%)
                # At least 20 samples for dev
                dev_size = max(20, len(fold_train_val) // 4)
                np.random.seed(self.random_state + fold_idx)
                dev_indices = np.random.choice(
                    len(fold_train_val), size=dev_size, replace=False)

                dev_mask = np.zeros(len(fold_train_val), dtype=bool)
                dev_mask[dev_indices] = True

                fold_dev = fold_train_val[dev_mask]
                fold_train = fold_train_val[~dev_mask]

                splits[f'fold_{fold_idx}'] = {
                    'train': fold_train['id'].tolist(),
                    'dev': fold_dev['id'].tolist(),
                    'test': fold_test['id'].tolist(),
                    'role': 'optimization'
                }

                fold_stats.append({
                    'fold': fold_idx,
                    'role': 'optimization',
                    'train_size': len(fold_train),
                    'dev_size': len(fold_dev),
                    'test_size': len(fold_test),
                    'train_question_types': fold_train['question_type_label'].value_counts().to_dict(),
                    'dev_question_types': fold_dev['question_type_label'].value_counts().to_dict(),
                    'test_question_types': fold_test['question_type_label'].value_counts().to_dict(),
                    'train_answer_bins': self.create_answer_count_bins().loc[fold_train.index].value_counts().to_dict(),
                    'dev_answer_bins': self.create_answer_count_bins().loc[fold_dev.index].value_counts().to_dict(),
                    'test_answer_bins': self.create_answer_count_bins().loc[fold_test.index].value_counts().to_dict()
                })

        return {
            'splits': splits,
            'statistics': fold_stats,
            'metadata': {
                'n_folds': n_folds,
                'optimization_folds': n_folds - 1,
                'final_test_fold': n_folds - 1,
                'total_samples': len(filtered_df),
                'random_state': self.random_state,
                'stratification_groups': len(valid_strats),
                'answer_count_bins': ['low (1-3)', 'medium (4-6)', 'high (7+)'],
                'creation_timestamp': pd.Timestamp.now().isoformat()
            }
        }
