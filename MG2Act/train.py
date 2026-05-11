import argparse
from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time
import math
import pandas as pd
from .dataset import MG2ActDataset, collate_samples
from .model import MG2ActModel
from .evaluate import evaluate
from .result_record import save_training_meta, save_detailed_results_with_metrics
import random

# Remove duplicate function definitions
def train_fixed_split(train_csv, val_csv, test_csv, args, device):
    """Train using fixed train/val/test files"""
    # Reduce print output during initialization
    if not args.quiet:
        print(f"\n{'='*80}")
        print(f"Starting training (fixed split)")
        print(f"Train: {train_csv}")
        print(f"Val: {val_csv}")
        print(f"Test: {test_csv}")
        print(f"{'='*80}")
    
    # Load datasets
    train_ds = MG2ActDataset(
        train_csv,
        col_activity="Score"
    )
    val_ds = MG2ActDataset(
        val_csv,
        col_activity="Score"
    )
    test_ds = MG2ActDataset(
        test_csv,
        col_activity="Score"
    )
    
    tr_loader = DataLoader(train_ds, batch_size=args.batch_size, 
                          shuffle=True, collate_fn=collate_samples)
    va_loader = DataLoader(val_ds, batch_size=args.batch_size, 
                          shuffle=False, collate_fn=collate_samples)
    te_loader = DataLoader(test_ds, batch_size=args.batch_size, 
                          shuffle=False, collate_fn=collate_samples)
    
    if not args.quiet:
        print(f"Train: {len(train_ds)} samples")
        print(f"Val: {len(val_ds)} samples")
        print(f"Test: {len(test_ds)} samples\n")
    
    # Initialize model
    # Default custom functional groups (user-provided)
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
    
    # Optimizer configuration
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    crit_train = torch.nn.MSELoss(reduction='none')
    crit_eval = torch.nn.MSELoss(reduction='mean')

    # Cosine Annealing Scheduler
    scheduler = None
    if args.use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # Training loop
    best_val = float("inf")
    best_val_mae = float("inf")
    best_test_mae = float("inf")
    best_epoch = 0
    patience_counter = 0
    threshold = 0.5  # Default threshold for compatibility
    start_time = time.time()

    # Concise determinism confirmation
    if not args.quiet:
        print(f"✓ Random seed set to: {args.seed}")
    
    # Overall training progress bar
    epoch_pbar = tqdm(range(1, args.epochs + 1), desc="Training progress", unit="epoch", disable=args.quiet)
    
    for ep in epoch_pbar:
        model.train()
        total = 0.0
        n = 0

        # Training phase: calculate train_loss only
        for batch in tr_loader:
            optim.zero_grad()
            e3 = batch["e3_seqs"]
            tgt = batch["target_seqs"]
            mol = batch["mol_batch"]
            y = batch["Score"].to(device)

            if y.dim() == 0:
                continue
            elif y.dim() > 1:
                y = y.squeeze()

            if len(e3) == 1 and y.size(0) != 1:
                y = y[:1]

            pred = model(e3, tgt, mol).view(-1)
            y = y.view(-1)

            if pred.shape[0] ! = y.shape[0]:
                raise RuntimeError("Severe dimension mismatch error! The number of predicted values ({pred.shape[0]}) does not match the number of labels ({y.shape[0]}).") )

            loss_vec = crit_train(pred, y)
            loss = loss_vec.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

            if scheduler is not None:
                scheduler.step()

            bs = y.size(0) if y.dim() > 0 else 1
            total += loss.item() * bs
            n += bs

        train_loss = total / max(1, n)

        # Validation phase: calculate val_loss and val_mae only (for early stopping and best model tracking)
        val_loss, val_mae = evaluate(model, va_loader, device, crit_eval, return_full=False)


        current_lr = optim.param_groups[0]['lr']

        # Update progress bar display info
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_mae': f'{val_mae:.4f}',
            'lr': f'{current_lr:.2e}'
        })

        # Print only at key points (every 10 epochs or when best model updates)
        if ep % 10 == 0 or val_loss < best_val:
            epoch_pbar.write(
                f"Epoch {ep}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_mae={val_mae:.4f}  lr={current_lr:.2e}"
            )
        
        # Save best model (based on val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_val_mae = val_mae
            best_epoch = ep
            patience_counter = 0

            # Calculate test_mae only when val_loss improves (fastest mode)
            best_test_mae = evaluate(model, te_loader, device, crit_eval, return_full=False, mae_only=True)

            out_dir = Path(args.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            ckpt = out_dir / "mg2act_best.pt"
            config = {
                "embed_dim": args.embed_dim,
                "attn_heads": args.attn_heads,
                "decoder_layers": args.decoder_layers,
                "mlp_hidden": args.mlp_hidden,
                "dropout": args.dropout,
                "proj_method": args.proj_method,
                "gnn_type": args.gnn_type,
                "gnn_layers": args.gnn_layers,
                "gnn_hidden_dim": args.gnn_hidden_dim,
                "use_multiscale_attention": args.use_multiscale_attention,
                "enable_fg_boost": args.enable_fg_boost,
                "fusion_method": args.fusion_method,
                "seed": args.seed,
            }
            torch.save({"model": model.state_dict(), "config": config}, ckpt)
        else:
            patience_counter += 1
            if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
                epoch_pbar.write(f"  ⚠️ Early stopping triggered: {args.early_stop_patience} consecutive epochs without val loss improvement")
                break
    
    epoch_pbar.close()
    total_time = time.time() - start_time
    print(f"\nTraining complete:")
    print(f"  Best val loss: {best_val:.4f}")
    print(f"  Best val MAE: {best_val_mae:.4f}")
    print(f"  Corresponding test MAE: {best_test_mae:.4f}")
    print(f"  Training time: {total_time/60:.2f} minutes")

    # Save training metadata
    save_training_meta(args, best_val, best_val_mae, best_test_mae, ep, best_epoch, total_time, threshold)
    
    return best_val, best_val_mae, best_test_mae, total_time, threshold


def main():
    parser = argparse.ArgumentParser(description="MG2Act fixed split training script")
    parser.add_argument("--folder", type=str, required=True, help="Folder path containing train.csv, val.csv, test.csv")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda:0")
    parser.add_argument("--out", type=str, default="outputs", help="Output directory")
    
    # Model parameters
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--attn_heads", type=int, default=1, help="Number of attention heads")
    parser.add_argument("--decoder_layers", type=int, default=2, help="Number of Transformer Decoder layers")
    parser.add_argument("--mlp_hidden", type=str, default="128,64", help="MLP hidden layer dimensions")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--proj_method", type=str, default="conv", choices=["conv", "mlp", "linear"],
                        help="Protein projection method: conv, mlp, linear")
    # Molecular encoder parameters (for building molecular graphs and extracting functional groups)
    parser.add_argument("--gnn_type", type=str, default="gcn", choices=["gcn", "gat"],
                        help="GNN type: gcn or gat")
    parser.add_argument("--gnn_layers", type=int, default=3, help="Number of GNN layers")
    parser.add_argument("--gnn_hidden_dim", type=int, default=128, help="GNN hidden dimension")

    # Attention configuration (fixed to True, as we only use functional group attention)
    parser.add_argument("--use_multiscale_attention", action="store_true", default=True,
                        help="Use multiscale attention (fixed to True)")

    # Optimization parameters
    parser.add_argument("--use_cosine", action="store_true", help="Enable cosine annealing learning rate scheduling")
    parser.add_argument("--early_stop_patience", type=int, default=0, help="Early stopping patience epochs (0=disabled)")

    # Fusion method parameters
    parser.add_argument("--fusion_method", type=str, default="attention", choices=["attention", "concat"],
                        help="Feature fusion method: attention (attention mechanism) or concat (direct concatenation for ablation)")

    # Functional group parameters (functional group attention only)
    parser.add_argument("--no_fg_boost", action="store_false", dest="enable_fg_boost", default=True,
                        help="Disable functional group boosting (do not use 2.0x weight for target groups, for ablation)")

    # Random seed parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default 42)")

    # Output control parameters
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output information (show only key information)")
    
    args = parser.parse_args()
    
    # Set quiet flag (default to False if not explicitly set)
    if not hasattr(args, 'quiet'):
        args.quiet = False
    device = torch.device(args.device)

    # Set random seed (ensure basic reproducibility)
    def setup_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        import os
        os.environ['PYTHONHASHSEED'] = str(seed)
    setup_seed(args.seed)

    if not args.quiet:
        print(f"✓ Random seed set to: {args.seed}")
    
    # Create output directory
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # Check folder and files
    folder = Path(args.folder)
    train_csv = folder / "train.csv"
    val_csv = folder / "val.csv"
    test_csv = folder / "test.csv"
    
    for f in [train_csv, val_csv, test_csv]:
        if not f.exists():
            print(f"Error: File not found: {f}")
            return
    
    if not args.quiet:
        print(f"\n{'='*80}")
        print(f"MG2Act Fixed Split Training")
        print(f"{'='*80}")
        print(f"Data folder: {args.folder}")
        print(f"Device: {device}")
        print(f"Output dir: {args.out}")
        print(f"Architecture config:")
        print(f"  - Embed dim: {args.embed_dim}")
        print(f"  - Attention heads: {args.attn_heads}")
        print(f"  - Decoder layers: {args.decoder_layers}")
        print(f"  - Protein projection: {args.proj_method}")
        print(f"  - Dropout: {args.dropout}")
        print(f"  - Functional group boost: {'Enabled (2.0x weight)' if args.enable_fg_boost else 'Disabled (1x weight)'}")
        print(f"  - Fusion method: {args.fusion_method}")
        print(f"{'='*80}\n")
    
    # Training
    best_val, best_val_mae, best_test_mae, total_time, threshold = train_fixed_split(
        train_csv, val_csv, test_csv, args, device
    )

    # Save detailed training results (including complete regression metrics)
    save_detailed_results_with_metrics(train_csv, val_csv, test_csv, args, device)
    


if __name__ == "__main__":
    main()
