import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from pathlib import Path
from typing import Dict, List
import ast


class StratifiedRAGDatasetSplitter:
    """
    Stratified splits for RAG at the QUESTION level.
    Stratification key: question_type × primary_tag_category_grouped × answer_count_bin.
    """

    def __init__(self, dataset_path: str, random_state: int = 42):
        self.dataset_path = Path(dataset_path)
        self.random_state = random_state
        self.questions_df: pd.DataFrame = None
        self._answer_bins: pd.Series = None
        self._strat_key: pd.Series = None

    def load_dataset(self):
        questions_file = self.dataset_path / "question.csv"  # NOTE: CSV, not JSONL
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
                self.questions_df['tags_parsed'] = [[]
                                                    for _ in range(len(self.questions_df))]

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
        cats = self.questions_df['primary_tag_category'].value_counts().head(
            top_k_categories).index
        grouped = self.questions_df['primary_tag_category'].apply(
            lambda x: x if x in cats else 'Other')
        bins = self._create_answer_count_bins()
        self._answer_bins = bins
        strat = (
            self.questions_df['question_type'].astype(str) + "_" +
            grouped.astype(str) + "_" +
            bins.astype(str)
        )
        return strat

    def create_cv_splits(self, n_folds: int = 5, min_samples_per_fold: int = 3) -> Dict:
        if self.questions_df is None:
            self.load_dataset()

        strat_key = self._create_strat_key()
        self._strat_key = strat_key

        # Keep strata with enough samples; relax threshold to avoid over-filtering
        needed = n_folds * min_samples_per_fold
        counts = strat_key.value_counts()
        valid = counts[counts >= needed].index
        mask = strat_key.isin(valid)
        filtered = self.questions_df[mask].copy()
        strat_f = strat_key[mask]
        bins_f = self._answer_bins[mask]

        print(
            f"Strata total: {len(counts)} | valid: {len(valid)} | kept samples: {len(filtered)}")

        id_col = 'question_id' if 'question_id' in filtered.columns else 'id'

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=self.random_state)
        splits = {}
        fold_stats = []
        rng = np.random.RandomState(self.random_state)

        for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(filtered, strat_f)):
            train_val_df = filtered.iloc[train_val_idx]
            test_df = filtered.iloc[test_idx]
            strat_train_val = strat_f.iloc[train_val_idx]

            if fold_idx == n_folds - 1:
                splits[f'fold_{fold_idx}'] = {
                    'train': [],
                    'dev': [],
                    'test': test_df[id_col].tolist(),
                    'role': 'final_test'
                }
                fold_stats.append({
                    'fold': fold_idx,
                    'role': 'final_test',
                    'train_size': 0,
                    'dev_size': 0,
                    'test_size': len(test_df),
                    'test_question_types': test_df['question_type'].value_counts().to_dict(),
                    'test_answer_bins': bins_f.loc[test_df.index].value_counts().to_dict()
                })
            else:
                # Stratified dev split inside train_val (e.g., 25% dev)
                sss = StratifiedShuffleSplit(
                    n_splits=1, test_size=0.25, random_state=self.random_state + fold_idx)
                (train_idx_rel, dev_idx_rel), = sss.split(
                    train_val_df, strat_train_val)

                fold_train = train_val_df.iloc[train_idx_rel]
                fold_dev = train_val_df.iloc[dev_idx_rel]

                splits[f'fold_{fold_idx}'] = {
                    'train': fold_train[id_col].tolist(),
                    'dev': fold_dev[id_col].tolist(),
                    'test': test_df[id_col].tolist(),
                    'role': 'optimization'
                }

                fold_stats.append({
                    'fold': fold_idx,
                    'role': 'optimization',
                    'train_size': len(fold_train),
                    'dev_size': len(fold_dev),
                    'test_size': len(test_df),
                    'train_question_types': fold_train['question_type'].value_counts().to_dict(),
                    'dev_question_types': fold_dev['question_type'].value_counts().to_dict(),
                    'test_question_types': test_df['question_type'].value_counts().to_dict(),
                    'train_answer_bins': bins_f.loc[fold_train.index].value_counts().to_dict(),
                    'dev_answer_bins': bins_f.loc[fold_dev.index].value_counts().to_dict(),
                    'test_answer_bins': bins_f.loc[test_df.index].value_counts().to_dict()
                })

        return {
            'splits': splits,
            'statistics': fold_stats,
            'metadata': {
                'n_folds': n_folds,
                'optimization_folds': n_folds - 1,
                'final_test_fold': n_folds - 1,
                'total_samples': int(len(filtered)),
                'random_state': self.random_state,
                'stratification_groups': int(len(valid)),
                'answer_count_bins': ['none (0)', 'low (1-3)', 'medium (4-6)', 'high (7+)'],
                'creation_timestamp': pd.Timestamp.now().isoformat()
            }
        }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Stratified splitting on SOSum StackOverflow dataset (CSV).")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to dataset root (expects question.csv)")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "dev", "test"])
    args = parser.parse_args()

    splitter = StratifiedRAGDatasetSplitter(args.dataset_path)
    splitter.load_dataset()
    cv_info = splitter.create_cv_splits()

    fold_key = f"fold_{args.fold}"
    split_ids = cv_info['splits'][fold_key][args.split]
    print(
        f"\nFold {args.fold} | Split: {args.split} | #Questions: {len(split_ids)}")
    for ext_id in split_ids[:20]:
        print(f"Question ID: {ext_id}")
