"""
消融实验专用结果记录模块
支持蛋白质编码消融实验的结果保存和评估

包含功能：
- 保存训练元信息
- 保存完整训练结果
- 评估回归指标
- 保存详细的回归指标结果
"""

import json
from pathlib import Path
import torch
import numpy as np
import math


def save_training_meta(args, best_val, best_val_mae, best_test_mae, ep, best_epoch, total_time, threshold):
    """保存训练元信息到results_meta.json"""
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
    print(f"✓ 训练元信息已写入: {Path(args.out) / 'results_meta.json'}")


def save_training_results(train_csv, val_csv, test_csv, best_val, best_val_mae, best_test_mae,
                         threshold, args):
    """保存完整训练结果到results.json"""
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
            # 消融实验特有参数
            "protein_method": getattr(args, 'protein_method', None),
            "protein_proj_method": getattr(args, 'protein_proj_method', None),
            "gnn_type": args.gnn_type,
            "gnn_layers": args.gnn_layers,
            "gnn_hidden_dim": args.gnn_hidden_dim,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "early_stop_patience": getattr(args, 'early_stop_patience', 0),
            "enable_fg_boost": args.enable_fg_boost,
            "fusion_method": args.fusion_method,
            "seed": args.seed,
        }
    }

    result_file = Path(args.out) / "results.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✓ 结果已保存到: {result_file}")
    print(f"✓ 最佳模型已保存到: {args.out}/mg2act_best.pt\n")


def evaluate_regression_metrics(model, loader, device):
    """评估回归指标（MAE、RMSE、R²）"""
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

    # 计算指标
    mae = np.mean(np.abs(y_pred - y_true))
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = math.sqrt(mse)

    # R²计算
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
    """保存包含详细回归指标的完整训练结果"""

    # 重新创建数据集（用于评估）
    from ..dataset import MG2ActDataset, collate_samples
    from torch.utils.data import DataLoader

    train_ds = MG2ActDataset(train_csv, col_activity="Score", col_target_name="PrimaryTarget")
    val_ds = MG2ActDataset(val_csv, col_activity="Score", col_target_name="PrimaryTarget")
    test_ds = MG2ActDataset(test_csv, col_activity="Score", col_target_name="PrimaryTarget")

    tr_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_samples)
    va_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_samples)
    te_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_samples)

    # 加载最佳模型
    ckpt = Path(args.out) / "mg2act_best.pt"
    checkpoint = torch.load(ckpt, map_location=device)

    # 重新创建模型（使用消融实验模型）
    from .model_ablation import MG2ActAblationModel
    default_custom_fg = {
        'cyclic_imide_1': 'C1(=O)CCCC(=O)N1',
        'cyclic_imide_2': 'C1(=O)CCNC(=O)N1',
        'cyclic_imide_3': 'N1C(=O)CCC1=O',
    }

    model = MG2ActAblationModel(
        device=device,
        embed_dim=args.embed_dim,
        attn_heads=args.attn_heads,
        decoder_layers=args.decoder_layers,
        mlp_hidden=args.mlp_hidden,
        dropout=args.dropout,
        protein_method=args.protein_method,
        protein_proj_method=getattr(args, 'protein_proj_method', 'conv'),
        gnn_type=args.gnn_type,
        gnn_layers=args.gnn_layers,
        gnn_hidden_dim=args.gnn_hidden_dim,
        enable_fg_boost=args.enable_fg_boost,
        fusion_method=args.fusion_method,
        custom_functional_groups=default_custom_fg,
    ).to(device)
    model.load_state_dict(checkpoint["model"])

    # 评估各数据集的回归指标
    print("\n评估详细回归指标...")
    train_metrics = evaluate_regression_metrics(model, tr_loader, device)
    val_metrics = evaluate_regression_metrics(model, va_loader, device)
    test_metrics = evaluate_regression_metrics(model, te_loader, device)

    # 打印结果
    print(f"\n{'='*80}")
    print("详细回归评估结果")
    print(f"{'='*80}")
    print(f"训练集 ({train_metrics['n_samples']} 样本):")
    print(f"  MAE:  {train_metrics['mae']:.4f}")
    print(f"  RMSE: {train_metrics['rmse']:.4f}")
    print(f"  R²:   {train_metrics['r2']:.4f}")
    print(f"\n验证集 ({val_metrics['n_samples']} 样本):")
    print(f"  MAE:  {val_metrics['mae']:.4f}")
    print(f"  RMSE: {val_metrics['rmse']:.4f}")
    print(f"  R²:   {val_metrics['r2']:.4f}")
    print(f"\n测试集 ({test_metrics['n_samples']} 样本):")
    print(f"  MAE:  {test_metrics['mae']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  R²:   {test_metrics['r2']:.4f}")
    print(f"{'='*80}\n")

    # 保存到results.json
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
            # 消融实验特有参数
            "protein_method": getattr(args, 'protein_method', None),
            "protein_proj_method": getattr(args, 'protein_proj_method', None),
            "gnn_type": args.gnn_type,
            "gnn_layers": args.gnn_layers,
            "gnn_hidden_dim": args.gnn_hidden_dim,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "early_stop_patience": getattr(args, 'early_stop_patience', 0),
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

    print(f"✓ 详细结果已保存到: {result_file}")
    print(f"✓ 最佳模型已保存到: {args.out}/mg2act_best.pt\n")
