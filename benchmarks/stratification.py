import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from typing import Dict, List
import ast


class StratifiedRAGDatasetSplitter:
    """
    Optimized stratified splits for RAG hyperparameter optimization.

    Strategy:
    - N-1 optimization folds: each uses 80% of data for evaluation
    - 1 final test fold: uses 20% of data for final validation
    - No test split within optimization folds (maximizes data usage)
    - Stratification: question_type × primary_tag_category × answer_count_bin
    """

    def __init__(self, dataset_path: str, random_state: int = 42):
        self.dataset_path = Path(dataset_path)
        self.random_state = random_state
        self.questions_df: pd.DataFrame = None
        self._answer_bins: pd.Series = None
        self._strat_key: pd.Series = None

    def load_dataset(self):
        """Φόρτωση και προεπεξεργασία του dataset ερωτημάτων."""
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
        """Κατηγοριοποίηση του αριθμού απαντήσεων σε bins."""
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
        Δημιουργία πολυδιάστατου stratification key.

        Το κλειδί συνδυάζει τρεις διαστάσεις για διασφάλιση αντιπροσωπευτικότητας:
        - question_type: Τύπος ερωτήματος
        - primary_tag_category: Κύρια θεματική κατηγορία (grouped)
        - answer_count_bin: Κατηγορία πλήθους απαντήσεων
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

    def create_cv_splits(self, n_folds: int = 5, min_samples_per_fold: int = 3) -> Dict:
        """
        Δημιουργία βελτιστοποιημένων stratified CV splits.

        Μεθοδολογία:
        - N-1 optimization folds: Κάθε fold χρησιμοποιεί 80% των δεδομένων
        - 1 final test fold: Χρησιμοποιεί το υπόλοιπο 20% για τελική επικύρωση

        Το βασικό πλεονέκτημα αυτής της προσέγγισης είναι η μέγιστη αξιοποίηση 
        των δεδομένων για την αξιολόγηση υπερπαραμέτρων, αφού δεν πραγματοποιείται 
        εκπαίδευση παραμέτρων μοντέλου.

        Args:
            n_folds: Συνολικός αριθμός folds (default: 5)
            min_samples_per_fold: Ελάχιστο πλήθος samples ανά stratum (default: 3)

        Returns:
            Dictionary με splits, statistics και metadata
        """
        if self.questions_df is None:
            self.load_dataset()

        strat_key = self._create_strat_key()
        self._strat_key = strat_key

        # Φιλτράρισμα strata με επαρκές μέγεθος δείγματος
        needed = n_folds * min_samples_per_fold
        counts = strat_key.value_counts()
        valid = counts[counts >= needed].index
        mask = strat_key.isin(valid)
        filtered = self.questions_df[mask].copy()
        strat_f = strat_key[mask]
        bins_f = self._answer_bins[mask]

        print(f"\n{'=' * 70}")
        print("STRATIFIED CROSS-VALIDATION SPLITS")
        print(f"{'=' * 70}")
        print(f"Total strata identified: {len(counts)}")
        print(f"Valid strata (≥{needed} samples): {len(valid)}")
        print(f"Samples retained: {len(filtered)} / {len(self.questions_df)}")
        print(f"{'=' * 70}\n")

        id_col = 'question_id' if 'question_id' in filtered.columns else 'id'

        # Δημιουργία stratified K-fold splits
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=self.random_state)
        splits = {}
        fold_stats = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(filtered, strat_f)):
            train_df = filtered.iloc[train_idx]  # 80% του συνόλου
            test_df = filtered.iloc[test_idx]    # 20% του συνόλου

            if fold_idx == n_folds - 1:
                # FINAL TEST FOLD: Μόνο test data για την τελική επικύρωση
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

                print(f"Fold {fold_idx} [FINAL TEST]:")
                print(
                    f"  Test size: {len(test_df)} ({len(test_df) / len(filtered) * 100:.1f}%)")

            else:
                # OPTIMIZATION FOLD: Όλο το 80% διατίθεται για αξιολόγηση
                # Τα train και dev είναι κενά - όλα τα data θα χρησιμοποιηθούν μαζί
                # Στο optimization script, αυτά θα συγχωνευτούν ούτως ή άλλως
                splits[f'fold_{fold_idx}'] = {
                    'train': train_df[id_col].tolist(),  # Όλο το 80%
                    'dev': [],  # Κενό - δεν χρειάζεται train/dev split
                    'test': [],  # Κενό - δεν χρειάζεται test split εδώ
                    'role': 'optimization'
                }

                fold_stats.append({
                    'fold': fold_idx,
                    'role': 'optimization',
                    'train_size': len(train_df),  # Όλο το διαθέσιμο 80%
                    'dev_size': 0,
                    'test_size': 0,
                    'train_question_types': train_df['question_type'].value_counts().to_dict(),
                    'dev_question_types': {},
                    'test_question_types': {},
                    'train_answer_bins': bins_f.loc[train_df.index].value_counts().to_dict(),
                    'dev_answer_bins': {},
                    'test_answer_bins': {}
                })

                print(f"Fold {fold_idx} [OPTIMIZATION]:")
                print(
                    f"  Evaluation size: {len(train_df)} ({len(train_df) / len(filtered) * 100:.1f}%)")

        print(f"\n{'=' * 70}")
        print("SUMMARY")
        print(f"{'=' * 70}")
        print(f"Optimization folds: {n_folds - 1}")
        print(f"Evaluation data per fold: ~{len(train_df)} samples (80%)")
        print(f"Final test fold: 1")
        print(f"Final test data: {len(test_df)} samples (20%)")
        print(
            f"Total data usage for optimization: {(n_folds - 1) * len(train_df)} evaluations")
        print(f"{'=' * 70}\n")

        return {
            'splits': splits,
            'statistics': fold_stats,
            'metadata': {
                'n_folds': n_folds,
                'optimization_folds': n_folds - 1,
                'final_test_fold': n_folds - 1,
                'total_samples': int(len(filtered)),
                'samples_per_optimization_fold': int(len(train_df)),
                'samples_final_test': int(len(test_df)),
                'random_state': self.random_state,
                'stratification_groups': int(len(valid)),
                'answer_count_bins': ['none (0)', 'low (1-3)', 'medium (4-6)', 'high (7+)'],
                'optimization_strategy': 'full_fold_evaluation',
                'note': 'Each optimization fold uses 80% of data; no internal train/dev split',
                'creation_timestamp': pd.Timestamp.now().isoformat()
            }
        }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Optimized stratified splitting for RAG hyperparameter tuning.")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to dataset root (expects question.csv)")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of folds (default: 5)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for CV splits JSON (optional)")
    args = parser.parse_args()

    splitter = StratifiedRAGDatasetSplitter(args.dataset_path)
    splitter.load_dataset()
    cv_info = splitter.create_cv_splits(n_folds=args.n_folds)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cv_info, f, ensure_ascii=False, indent=2)
        print(f"\n✓ CV splits saved to: {output_path}")

    # Εμφάνιση στατιστικών
    print("\nFold Statistics:")
    for stat in cv_info['statistics']:
        fold = stat['fold']
        role = stat['role']
        if role == 'optimization':
            print(
                f"  Fold {fold}: {stat['train_size']} samples for evaluation")
        else:
            print(f"  Fold {fold}: {stat['test_size']} samples for final test")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from typing import Dict, List
import ast


class StratifiedRAGDatasetSplitter:
    """
    Optimized stratified splits for RAG hyperparameter optimization.

    Strategy:
    - N-1 optimization folds: each uses 80% of data for evaluation
    - 1 final test fold: uses 20% of data for final validation
    - No test split within optimization folds (maximizes data usage)
    - Stratification: question_type × primary_tag_category × answer_count_bin
    """

    def __init__(self, dataset_path: str, random_state: int = 42):
        self.dataset_path = Path(dataset_path)
        self.random_state = random_state
        self.questions_df: pd.DataFrame = None
        self._answer_bins: pd.Series = None
        self._strat_key: pd.Series = None

    def load_dataset(self):
        """Φόρτωση και προεπεξεργασία του dataset ερωτημάτων."""
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
        """Κατηγοριοποίηση του αριθμού απαντήσεων σε bins."""
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
        Δημιουργία πολυδιάστατου stratification key.

        Το κλειδί συνδυάζει τρεις διαστάσεις για διασφάλιση αντιπροσωπευτικότητας:
        - question_type: Τύπος ερωτήματος
        - primary_tag_category: Κύρια θεματική κατηγορία (grouped)
        - answer_count_bin: Κατηγορία πλήθους απαντήσεων
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

    def create_cv_splits(self, n_folds: int = 5, min_samples_per_fold: int = 3) -> Dict:
        """
        Δημιουργία βελτιστοποιημένων stratified CV splits.

        Μεθοδολογία:
        - N-1 optimization folds: Κάθε fold χρησιμοποιεί 80% των δεδομένων
        - 1 final test fold: Χρησιμοποιεί το υπόλοιπο 20% για τελική επικύρωση

        Το βασικό πλεονέκτημα αυτής της προσέγγισης είναι η μέγιστη αξιοποίηση 
        των δεδομένων για την αξιολόγηση υπερπαραμέτρων, αφού δεν πραγματοποιείται 
        εκπαίδευση παραμέτρων μοντέλου.

        Args:
            n_folds: Συνολικός αριθμός folds (default: 5)
            min_samples_per_fold: Ελάχιστο πλήθος samples ανά stratum (default: 3)

        Returns:
            Dictionary με splits, statistics και metadata
        """
        if self.questions_df is None:
            self.load_dataset()

        strat_key = self._create_strat_key()
        self._strat_key = strat_key

        # Φιλτράρισμα strata με επαρκές μέγεθος δείγματος
        needed = n_folds * min_samples_per_fold
        counts = strat_key.value_counts()
        valid = counts[counts >= needed].index
        mask = strat_key.isin(valid)
        filtered = self.questions_df[mask].copy()
        strat_f = strat_key[mask]
        bins_f = self._answer_bins[mask]

        print(f"\n{'=' * 70}")
        print("STRATIFIED CROSS-VALIDATION SPLITS")
        print(f"{'=' * 70}")
        print(f"Total strata identified: {len(counts)}")
        print(f"Valid strata (≥{needed} samples): {len(valid)}")
        print(f"Samples retained: {len(filtered)} / {len(self.questions_df)}")
        print(f"{'=' * 70}\n")

        id_col = 'question_id' if 'question_id' in filtered.columns else 'id'

        # Δημιουργία stratified K-fold splits
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=self.random_state)
        splits = {}
        fold_stats = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(filtered, strat_f)):
            train_df = filtered.iloc[train_idx]  # 80% του συνόλου
            test_df = filtered.iloc[test_idx]    # 20% του συνόλου

            if fold_idx == n_folds - 1:
                # FINAL TEST FOLD: Μόνο test data για την τελική επικύρωση
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

                print(f"Fold {fold_idx} [FINAL TEST]:")
                print(
                    f"  Test size: {len(test_df)} ({len(test_df) / len(filtered) * 100:.1f}%)")

            else:
                # OPTIMIZATION FOLD: Όλο το 80% διατίθεται για αξιολόγηση
                # Τα train και dev είναι κενά - όλα τα data θα χρησιμοποιηθούν μαζί
                # Στο optimization script, αυτά θα συγχωνευτούν ούτως ή άλλως
                splits[f'fold_{fold_idx}'] = {
                    'train': train_df[id_col].tolist(),  # Όλο το 80%
                    'dev': [],  # Κενό - δεν χρειάζεται train/dev split
                    'test': [],  # Κενό - δεν χρειάζεται test split εδώ
                    'role': 'optimization'
                }

                fold_stats.append({
                    'fold': fold_idx,
                    'role': 'optimization',
                    'train_size': len(train_df),  # Όλο το διαθέσιμο 80%
                    'dev_size': 0,
                    'test_size': 0,
                    'train_question_types': train_df['question_type'].value_counts().to_dict(),
                    'dev_question_types': {},
                    'test_question_types': {},
                    'train_answer_bins': bins_f.loc[train_df.index].value_counts().to_dict(),
                    'dev_answer_bins': {},
                    'test_answer_bins': {}
                })

                print(f"Fold {fold_idx} [OPTIMIZATION]:")
                print(
                    f"  Evaluation size: {len(train_df)} ({len(train_df) / len(filtered) * 100:.1f}%)")

        print(f"\n{'=' * 70}")
        print("SUMMARY")
        print(f"{'=' * 70}")
        print(f"Optimization folds: {n_folds - 1}")
        print(f"Evaluation data per fold: ~{len(train_df)} samples (80%)")
        print(f"Final test fold: 1")
        print(f"Final test data: {len(test_df)} samples (20%)")
        print(
            f"Total data usage for optimization: {(n_folds - 1) * len(train_df)} evaluations")
        print(f"{'=' * 70}\n")

        return {
            'splits': splits,
            'statistics': fold_stats,
            'metadata': {
                'n_folds': n_folds,
                'optimization_folds': n_folds - 1,
                'final_test_fold': n_folds - 1,
                'total_samples': int(len(filtered)),
                'samples_per_optimization_fold': int(len(train_df)),
                'samples_final_test': int(len(test_df)),
                'random_state': self.random_state,
                'stratification_groups': int(len(valid)),
                'answer_count_bins': ['none (0)', 'low (1-3)', 'medium (4-6)', 'high (7+)'],
                'optimization_strategy': 'full_fold_evaluation',
                'note': 'Each optimization fold uses 80% of data; no internal train/dev split',
                'creation_timestamp': pd.Timestamp.now().isoformat()
            }
        }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Optimized stratified splitting for RAG hyperparameter tuning.")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to dataset root (expects question.csv)")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of folds (default: 5)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for CV splits JSON (optional)")
    args = parser.parse_args()

    splitter = StratifiedRAGDatasetSplitter(args.dataset_path)
    splitter.load_dataset()
    cv_info = splitter.create_cv_splits(n_folds=args.n_folds)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cv_info, f, ensure_ascii=False, indent=2)
        print(f"\n✓ CV splits saved to: {output_path}")

    # Εμφάνιση στατιστικών
    print("\nFold Statistics:")
    for stat in cv_info['statistics']:
        fold = stat['fold']
        role = stat['role']
        if role == 'optimization':
            print(
                f"  Fold {fold}: {stat['train_size']} samples for evaluation")
        else:
            print(f"  Fold {fold}: {stat['test_size']} samples for final test")
