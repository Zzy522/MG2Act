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
                     col_activity: str = "Score"): 
            super().__init__()
            df = pd.read_csv(csv_path)
                         
            needed = [col_e3, col_target, col_smiles]
            if col_activity in df.columns:
                needed.append(col_activity)
                
            for c in needed:
                if c not in df.columns:
                    raise ValueError(f"CSV Error: {c}")
                    
            df = df.dropna(subset=needed).reset_index(drop=True)
    
            if col_activity not in df.columns:
                df[col_activity] = 0.0
            
            self.samples: List[Sample] = []
            for _, row in df.iterrows():
                try:
                    smi = str(row[col_smiles]).strip()
    
                    e3 = str(row[col_e3]).strip()
                    tgt = str(row[col_target]).strip()
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
