import torch
import numpy as np
import math
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

def evaluate(model, loader, device, crit, return_full=True, mae_only=False):
    """Evaluate model.

    Args:
        return_full: Whether to return full metrics (loss, mae, rmse, r2)
        mae_only: Whether to return only MAE (fastest mode for MAE-only scenarios)
    """
    if mae_only:
        # Fastest mode: Calculate MAE only
        model.eval()
        total_mae = 0.0
        n = 0

        with torch.no_grad():
            for batch in loader:
                e3 = batch["e3_seqs"]
                tgt = batch["target_seqs"]
                mol = batch["mol_batch"]
                y = batch["Score"].to(device).view(-1)

                pred = model(e3, tgt, mol).view(-1)

                if pred.shape[0] != y.shape[0]:
                    if pred.shape[0] ! = y.shape[0]:
                        raise RuntimeError("Severe dimension mismatch error! The number of predicted values ({pred.shape[0]}) does not match the number of labels ({y.shape[0]}).") )
                    # m = min(pred.shape[0], y.shape[0])
                    # pred = pred[:m]
                    # y = y[:m]

                mae = (pred - y).abs().mean()
                bs = y.numel()
                total_mae += mae.item() * bs
                n += bs

        return total_mae / max(1, n)

    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    n = 0

    # Calculate additional statistics only when full metrics needed
    if return_full:
        sse = 0.0
        sum_y = 0.0
        sum_y_sq = 0.0

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

            loss = crit(pred, y)
            if hasattr(loss, 'dim') and loss.dim() > 0:
                loss = loss.mean()
            mae = (pred - y).abs().mean()
            bs = y.numel()
            total_loss += loss.item() * bs
            total_mae += mae.item() * bs

            if return_full:
                sse += (pred - y).pow(2).sum().item()
                sum_y += y.sum().item()
                sum_y_sq += y.pow(2).sum().item()

            n += bs

    val_loss = total_loss / max(1, n)
    val_mae = total_mae / max(1, n)

    if return_full:
        val_rmse = math.sqrt(sse / max(1, n)) if n > 0 else float("nan")
        mean_y = (sum_y / n) if n > 0 else 0.0
        ss_tot = sum_y_sq - n * (mean_y ** 2)
        val_r2 = 1.0 - (sse / ss_tot) if ss_tot > 0 else float("nan")
        return val_loss, val_mae, val_rmse, val_r2
    else:
        return val_loss, val_mae

