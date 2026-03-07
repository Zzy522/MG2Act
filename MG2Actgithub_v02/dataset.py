from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter


@dataclass
class Sample:
    """Data sample (simplified: no physicochemical features)"""
    e3_seq: str
    target_seq: str
    smiles: str
    activity: float


def bayesian_shrinkage_target_norm(df, target_col, score_col, lambda_=10):
    """
    Bayesian shrinkage target normalization.
    Counteracts "target bias" by using global statistics when target has few samples.

    Args:
        df: DataFrame
        target_col: Target column name (e.g., PrimaryTarget)
        score_col: Score column name (e.g., Score)
        lambda_: Shrinkage strength, higher values favor global statistics

    Returns:
        DataFrame with standardized_score column
    """
    # Global mean and variance
    global_mean = df[score_col].mean()
    global_std = df[score_col].std()

    result_df = df.copy()
    result_df['standardized_score'] = np.nan

    for target, sub in df.groupby(target_col):
        n = len(sub)
        mu_t = sub[score_col].mean()
        sigma_t = sub[score_col].std()

        # Bayesian shrinkage: Shrink to global mean when few samples
        mu_t_shrink = (n / (n + lambda_)) * mu_t + (lambda_ / (n + lambda_)) * global_mean
        sigma_t_shrink = np.sqrt((n / (n + lambda_)) * sigma_t**2 + (lambda_ / (n + lambda_)) * global_std**2)

        # Standardization
        result_df.loc[sub.index, 'standardized_score'] = (sub[score_col] - mu_t_shrink) / sigma_t_shrink

    return result_df


class MG2ActDataset(Dataset):
    """
    MG2Act dataset (simplified version).

    Features:
    - Uses only sequence information (E3, Target, SMILES)
    - Removes physicochemical feature inputs (cLogP, etc.)
    - Returns SMILES strings directly (for Transformer molecular encoder)
    - Supports Bayesian shrinkage target normalization to counter target bias
    """
    def __init__(self, csv_path: str,
                 col_e3: str = "E3_seq",
                 col_target: str = "Target_seq",
                 col_smiles: str = "Molecule_SMILES",
                 col_activity: str = "Score",
                 # Bayesian shrinkage normalization parameters
                 col_target_name: str = "PrimaryTarget",  # Target name column
                 use_bayesian_norm: bool = False,  # Whether to use Bayesian shrinkage normalization
                 bayesian_lambda: float = 10.0):  # Bayesian shrinkage parameter
        super().__init__()
        df = pd.read_csv(csv_path)
        needed = [col_e3, col_target, col_smiles, col_activity]
        for c in needed:
            if c not in df.columns:
                raise ValueError(f"CSV缺少列: {c}")
        df = df.dropna(subset=needed).reset_index(drop=True)

        # Apply Bayesian shrinkage normalization (if enabled)
        if use_bayesian_norm:
            # Check if target name column exists
            if col_target_name not in df.columns:
                # Silent handling: only print warning when needed
                df["standardized_score"] = df[col_activity]
            else:
                df = bayesian_shrinkage_target_norm(df, col_target_name, col_activity, lambda_=bayesian_lambda)
                # Update activity column to standardized scores
                df[col_activity] = df['standardized_score']

        self.samples: List[Sample] = []
        for _, row in df.iterrows():
            try:
                e3 = str(row[col_e3]).strip()
                tgt = str(row[col_target]).strip()
                smi = str(row[col_smiles]).strip()
                y = float(row[col_activity])
                self.samples.append(Sample(e3, tgt, smi, y))
            except Exception:
                continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        return {
            "e3_seq": s.e3_seq,
            "target_seq": s.target_seq,
            "smiles": s.smiles,
            "Score": torch.tensor([s.activity], dtype=torch.float),
        }


def collate_samples(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collation function (simplified version).

    Args:
        items: List of data samples

    Returns:
        Batched dictionary
    """
    e3_seqs = [it["e3_seq"] for it in items]
    target_seqs = [it["target_seq"] for it in items]
    smiles_list = [it["smiles"] for it in items]
    y = torch.cat([it["Score"] for it in items], dim=0)
    return {
        "e3_seqs": e3_seqs,
        "target_seqs": target_seqs,
        "mol_batch": smiles_list,
        "Score": y,        # [B]
    }
