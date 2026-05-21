"""
Inference script: Given a trained model directory (containing mg2act_best.pt and results.json)
and a CSV with inference fields (must contain columns: E3_seq, Target_seq, Molecule_SMILES),
automatically parses architecture, loads weights, and outputs CSV with predictions.

Usage examples:
    python -m MG2Act.predict_infer \
        --input data/test.csv \
        --model_dir model_wt \
        --output predict/predictions.csv \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch

from .model import MG2ActModel


def _load_results_config(model_dir: Path) -> Dict:
    results_path = model_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found: {results_path}")
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    return results.get("config", {})


def _find_checkpoint(model_dir: Path) -> Path:
    ckpt = model_dir / "mg2act_best.pt"
    if ckpt.exists():
        return ckpt
    # Fallback: take first .pt file in directory
    pt_files = sorted(model_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt weights found in model directory: {model_dir}")
    return pt_files[0]


def load_model_and_config(model_dir: Path, device: torch.device) -> Tuple[MG2ActModel, Dict]:
    """
    Load results.json to infer architecture and load weights.
    """
    cfg = _load_results_config(model_dir)
    ckpt_path = _find_checkpoint(model_dir)

    model_kwargs = dict(
        device=device,
        embed_dim=cfg.get("embed_dim", 64),
        attn_heads=cfg.get("attn_heads", 4),
        decoder_layers=cfg.get("decoder_layers", 2),
        mlp_hidden=cfg.get("mlp_hidden", "128,64"),
        dropout=cfg.get("dropout", 0.1),
        proj_method=cfg.get("proj_method", "conv"),
        gnn_type=cfg.get("gnn_type", "gcn"),
        gnn_layers=cfg.get("gnn_layers", 3),
        gnn_hidden_dim=cfg.get("gnn_hidden_dim", 128),
        use_multiscale_attention=cfg.get("use_multiscale_attention", True),
        enable_fg_boost=cfg.get("enable_fg_boost", True),
        fusion_method=cfg.get("fusion_method", "attention"),
    )

    model = MG2ActModel(**model_kwargs).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"⚠️ Missing weight keys: {missing}")
    if unexpected:
        print(f"⚠️ Unexpected weight keys: {unexpected}")

    model.eval()
    return model, cfg


def read_infer_csv(csv_path: Path) -> pd.DataFrame:
    required_cols = ["E3_seq", "Target_seq", "Molecule_SMILES"]
    df = pd.read_csv(csv_path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    return df


@torch.inference_mode()
def run_inference(
    df: pd.DataFrame,
    model: MG2ActModel,
    batch_size: int,
    device: torch.device,
) -> List[float]:
    preds: List[float] = []
    total = len(df)
    for start in range(0, total, batch_size):
        batch = df.iloc[start : start + batch_size]
        e3 = batch["E3_seq"].tolist()
        tgt = batch["Target_seq"].tolist()
        smi = batch["Molecule_SMILES"].tolist()
        scores = model(e3, tgt, smi)  # [B]
        preds.extend(scores.detach().cpu().float().tolist())
    return preds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MG2Act inference script")
    parser.add_argument("--input", required=True, type=Path, help="Input CSV for inference, must contain E3_seq/Target_seq/Molecule_SMILES")
    parser.add_argument("--model_dir", required=True, type=Path, help="Training output directory containing mg2act_best.pt and results.json")
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"), help="Output path for predictions CSV")
    parser.add_argument("--batch_size", type=int, default=4, help="Inference batch size")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="cpu or cuda:0",
    )
    parser.add_argument("--quiet", action="store_true", help="no-log")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    if not args.quiet:
        print(f"✓ Using device: {device}")
        print(f"✓ Loading model dir: {args.model_dir}")
        print(f"✓ Loading data: {args.input}")

    df = read_infer_csv(args.input)
    if len(df) == 0:
        raise ValueError("Input CSV empty after removing missing values, cannot infer.")

    model, cfg = load_model_and_config(args.model_dir, device)

    if not args.quiet:
        print("✓ Auto-detected architecture:", {
            "fusion_method": cfg.get("fusion_method", "attention"),
            "gnn_type": cfg.get("gnn_type", "gcn"),
            "embed_dim": cfg.get("embed_dim", 64),
            "attn_heads": cfg.get("attn_heads", 4),
            "decoder_layers": cfg.get("decoder_layers", 2),
            "proj_method": cfg.get("proj_method", "conv"),
        })
        print(f"✓ Starting inference, {len(df)} samples, batch_size={args.batch_size}")

    preds = run_inference(df, model, args.batch_size, device)
    df_out = df.copy()
    df_out["Predicted_Score"] = preds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)

    if not args.quiet:
        print(f"✓ Complete, saved to: {args.output}")


if __name__ == "__main__":
    main()

