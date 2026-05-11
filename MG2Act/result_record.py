import json
from pathlib import Path
import torch
import numpy as np
import math

def save_training_meta(args, best_val, best_val_mae, best_test_mae, ep, best_epoch, total_time, threshold):
    """Save training metadata to results_meta.json"""
    meta = {
        "epochs_planned": int(args.epochs),
        "epochs_done": int(ep),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "best_val_mae": float(best_val_mae),
        "best_test_mae": float(best_test_mae),
        "training_time_min": round(total_time / 60.0, 2),
        "threshold_used": float(threshold),
    }
    with open(Path(args.out) / "results_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"✓ Training metadata saved to: {Path(args.out) / 'results_meta.json'}")

def save_training_results(train_csv, val_csv, test_csv, best_val, best_val_mae, best_test_mae,
                         threshold, args):
    """Save complete training results to results.json"""
    results = {
        "strategy": "fixed_split",
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "best_val_loss": float(best_val),
        "best_val_mae": float(best_val_mae),
        "best_test_mae": float(best_test_mae),
        "classification_threshold": float(threshold),
        "config": {
            "embed_dim": args.embed_dim,
            "attn_heads": args.attn_heads,
            "decoder_layers": args.decoder_layers,
            "mlp_hidden": args.mlp_hidden,
            "dropout": args.dropout,
            "proj_method": args.proj_method,
            "gnn_type": args.gnn_type,
            "gnn_layers": args.gnn_layers,
            "gnn_hidden_dim": args.gnn_hidden_dim,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "use_cosine": args.use_cosine,
            "early_stop_patience": args.early_stop_patience,
            "use_multiscale_attention": args.use_multiscale_attention,
            "enable_fg_boost": args.enable_fg_boost,
            "fusion_method": args.fusion_method,
            "seed": args.seed,
        }
    }

    result_file = Path(args.out) / "results.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✓ Results saved to: {result_file}")
    print(f"✓ Best model saved to: {args.out}/mg2act_best.pt\n")

def evaluate_regression_metrics(model, loader, device):
    """Evaluate regression metrics (MAE, RMSE, R²)"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            e3 = batch["e3_seqs"]
            tgt = batch["target_seqs"]
            mol = batch["mol_batch"]
            y = batch["Score"].to(device).view(-1)

            pred = model(e3, tgt, mol).view(-1)

            if pred.shape[0] != y.shape[0]:
                m = min(pred.shape[0], y.shape[0])
                pred = pred[:m]
                y = y[:m]

            all_preds.append(pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    # Calculate metrics
    mae = np.mean(np.abs(y_pred - y_true))
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = math.sqrt(mse)

    # Calculate R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "n_samples": len(y_true)
    }

def save_detailed_results_with_metrics(train_csv, val_csv, test_csv, args, device):
    """Save complete training results with detailed regression metrics"""

    # Recreate datasets for evaluation
    from .dataset import MG2ActDataset, collate_samples
    from torch.utils.data import DataLoader

    train_ds = MG2ActDataset(train_csv, col_activity="Score")
    val_ds = MG2ActDataset(val_csv, col_activity="Score")
    test_ds = MG2ActDataset(test_csv, col_activity="Score")

    tr_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_samples)
    va_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_samples)
    te_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_samples)

    # Load best model
    ckpt = Path(args.out) / "mg2act_best.pt"
    checkpoint = torch.load(ckpt, map_location=device)

    # Recreate model with same config as training
    from .model import MG2ActModel
    default_custom_fg = {
        'cyclic_imide_1': 'C1(=O)CCCC(=O)N1',
        'cyclic_imide_2': 'C1(=O)CCNC(=O)N1',
        'cyclic_imide_3': 'N1C(=O)CCC1=O',
    }

    model = MG2ActModel(
        device=device,
        embed_dim=args.embed_dim,
        attn_heads=args.attn_heads,
        decoder_layers=args.decoder_layers,
        mlp_hidden=args.mlp_hidden,
        dropout=args.dropout,
        proj_method=args.proj_method,
        gnn_type=args.gnn_type,
        gnn_layers=args.gnn_layers,
        gnn_hidden_dim=args.gnn_hidden_dim,
        custom_functional_groups=default_custom_fg,
        enable_fg_boost=args.enable_fg_boost,
        fusion_method=args.fusion_method,
    ).to(device)
    model.load_state_dict(checkpoint["model"])

    # Evaluate regression metrics for each dataset
    print("\nEvaluating detailed regression metrics...")
    train_metrics = evaluate_regression_metrics(model, tr_loader, device)
    val_metrics = evaluate_regression_metrics(model, va_loader, device)
    test_metrics = evaluate_regression_metrics(model, te_loader, device)

    # Print results
    print(f"\n{'='*80}")
    print("Detailed Regression Evaluation Results")
    print(f"{'='*80}")
    print(f"Train ({train_metrics['n_samples']} samples):")
    print(f"  MAE:  {train_metrics['mae']:.4f}")
    print(f"  RMSE: {train_metrics['rmse']:.4f}")
    print(f"  R²:   {train_metrics['r2']:.4f}")
    print(f"\nVal ({val_metrics['n_samples']} samples):")
    print(f"  MAE:  {val_metrics['mae']:.4f}")
    print(f"  RMSE: {val_metrics['rmse']:.4f}")
    print(f"  R²:   {val_metrics['r2']:.4f}")
    print(f"\nTest ({test_metrics['n_samples']} samples):")
    print(f"  MAE:  {test_metrics['mae']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  R²:   {test_metrics['r2']:.4f}")
    print(f"{'='*80}\n")

    # Save to results.json
    results = {
        "strategy": "fixed_split",
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "config": {
            "embed_dim": args.embed_dim,
            "attn_heads": args.attn_heads,
            "decoder_layers": args.decoder_layers,
            "mlp_hidden": args.mlp_hidden,
            "dropout": args.dropout,
            "proj_method": args.proj_method,
            "gnn_type": args.gnn_type,
            "gnn_layers": args.gnn_layers,
            "gnn_hidden_dim": args.gnn_hidden_dim,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "use_cosine": args.use_cosine,
            "early_stop_patience": args.early_stop_patience,
            "use_multiscale_attention": args.use_multiscale_attention,
            "enable_fg_boost": args.enable_fg_boost,
            "fusion_method": args.fusion_method,
            "seed": args.seed,
        },
        "regression_metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics
        }
    }

    result_file = Path(args.out) / "results.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✓ Detailed results saved to: {result_file}")
    print(f"✓ Best model saved to: {args.out}/mg2act_best.pt\n")
