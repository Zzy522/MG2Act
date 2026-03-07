"""
蛋白端ESMC消融实验训练脚本

支持不同的蛋白质编码方法：
1. ESMC - 使用预训练的ESMC编码器
2. ACC - 自协互协特征编码
3. OneHot - One-hot编码

使用方法：
python train_protein_ablation.py --folder splits/train_val_test \
    --protein_method esmc --out outputs_esmc

python train_protein_ablation.py --folder splits/train_val_test \
    --protein_method acc --out outputs_acc

python train_protein_ablation.py --folder splits/train_val_test \
    --protein_method onehot --out outputs_onehot
"""

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
from ..dataset import MG2ActDataset, collate_samples
from .model_ablation import MG2ActAblationModel
from ..evaluate import evaluate
from .result_record_ablation import save_training_meta, save_detailed_results_with_metrics
import random


# 移除重复的函数定义
def train_fixed_split(train_csv, val_csv, test_csv, args, device):
    """使用固定的train/val/test文件进行训练"""
    # 减少初始化时的print输出
    if not args.quiet:
        print(f"\n{'='*80}")
        print(f"开始蛋白端消融实验训练（固定拆分）")
        print(f"蛋白质编码方法: {args.protein_method}")
        print(f"训练集: {train_csv}")
        print(f"验证集: {val_csv}")
        print(f"测试集: {test_csv}")
        print(f"{'='*80}")

    # 加载数据集
    train_ds = MG2ActDataset(
        train_csv,
        col_activity="Score",
        col_target_name="PrimaryTarget"
    )
    val_ds = MG2ActDataset(
        val_csv,
        col_activity="Score",
        col_target_name="PrimaryTarget"
    )
    test_ds = MG2ActDataset(
        test_csv,
        col_activity="Score",
        col_target_name="PrimaryTarget"
    )

    tr_loader = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, collate_fn=collate_samples)
    va_loader = DataLoader(val_ds, batch_size=args.batch_size,
                          shuffle=False, collate_fn=collate_samples)
    te_loader = DataLoader(test_ds, batch_size=args.batch_size,
                          shuffle=False, collate_fn=collate_samples)

    if not args.quiet:
        print(f"训练集: {len(train_ds)} 样本")
        print(f"验证集: {len(val_ds)} 样本")
        print(f"测试集: {len(test_ds)} 样本\n")

    # 初始化模型
    # 默认自定义官能团（用户提供的）
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
        protein_proj_method=args.protein_proj_method,
        gnn_type=args.gnn_type,
        gnn_layers=args.gnn_layers,
        gnn_hidden_dim=args.gnn_hidden_dim,
        enable_fg_boost=args.enable_fg_boost,
        fusion_method=args.fusion_method,
        custom_functional_groups=default_custom_fg,
    ).to(device)

    # 优化器配置
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    crit_train = torch.nn.MSELoss(reduction='none')
    crit_eval = torch.nn.MSELoss(reduction='mean')

    # Cosine Annealing Scheduler
    scheduler = None
    if args.use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # 训练循环
    best_val = float("inf")
    best_val_mae = float("inf")
    best_test_mae = float("inf")
    best_epoch = 0
    patience_counter = 0
    threshold = 0.5  # 默认阈值，用于兼容性
    start_time = time.time()

    # 简洁的确定性确认
    if not args.quiet:
        print(f"✓ 随机种子设置为: {args.seed}")

    # 整体训练进度条
    epoch_pbar = tqdm(range(1, args.epochs + 1), desc="训练进度", unit="epoch", disable=args.quiet)

    for ep in epoch_pbar:
        model.train()
        total = 0.0
        n = 0

        # 训练阶段：只计算train_loss
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

            min_size = min(pred.shape[0], y.shape[0])
            pred = pred[:min_size]
            y = y[:min_size]

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

        # 验证阶段：只计算val_loss和val_mae（用于早停和记录最佳模型）
        val_loss, val_mae = evaluate(model, va_loader, device, crit_eval, return_full=False)

        current_lr = optim.param_groups[0]['lr']

        # 更新进度条显示信息
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_mae': f'{val_mae:.4f}',
            'lr': f'{current_lr:.2e}'
        })

        # 只在关键点打印（每10个epoch或最佳模型更新时）
        if ep % 10 == 0 or val_loss < best_val:
            epoch_pbar.write(
                f"Epoch {ep}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_mae={val_mae:.4f}  lr={current_lr:.2e}"
            )

        # 保存最佳模型（基于val_loss）
        if val_loss < best_val:
            best_val = val_loss
            best_val_mae = val_mae
            best_epoch = ep
            patience_counter = 0

            # 只在val_loss改善时计算一次test_mae（最快模式）
            best_test_mae = evaluate(model, te_loader, device, crit_eval, return_full=False, mae_only=True)

            out_dir = Path(args.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            ckpt = out_dir / "mg2act_best.pt"
            config = {
                "protein_method": args.protein_method,
                "embed_dim": args.embed_dim,
                "attn_heads": args.attn_heads,
                "decoder_layers": args.decoder_layers,
                "mlp_hidden": args.mlp_hidden,
                "dropout": args.dropout,
                "protein_proj_method": args.protein_proj_method,
                "gnn_type": args.gnn_type,
                "gnn_layers": args.gnn_layers,
                "gnn_hidden_dim": args.gnn_hidden_dim,
                "enable_fg_boost": args.enable_fg_boost,
                "fusion_method": args.fusion_method,
                "seed": args.seed,
            }
            torch.save({"model": model.state_dict(), "config": config}, ckpt)
        else:
            patience_counter += 1
            if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
                epoch_pbar.write(f"  ⚠️ 早停触发: 连续 {args.early_stop_patience} 轮验证损失未改善")
                break

    epoch_pbar.close()
    total_time = time.time() - start_time
    print(f"\n训练完成 ({args.protein_method}编码):")
    print(f"  最佳验证损失: {best_val:.4f}")
    print(f"  最佳验证 MAE: {best_val_mae:.4f}")
    print(f"  对应测试 MAE: {best_test_mae:.4f}")
    print(f"  训练时间: {total_time/60:.2f} 分钟")

    # 保存训练元信息
    save_training_meta(args, best_val, best_val_mae, best_test_mae, ep, best_epoch, total_time, threshold)

    return best_val, best_val_mae, best_test_mae, total_time, threshold


def main():
    parser = argparse.ArgumentParser(description="MG2Act 蛋白端消融实验训练脚本")
    parser.add_argument("--folder", type=str, required=True, help="包含train.csv、val.csv、test.csv的文件夹路径")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--device", type=str, default="cpu", help="设备: cpu 或 cuda:0")
    parser.add_argument("--out", type=str, default="outputs", help="输出目录")

    # 模型参数
    parser.add_argument("--embed_dim", type=int, default=64, help="嵌入维度")
    parser.add_argument("--attn_heads", type=int, default=4, help="注意力头数")
    parser.add_argument("--decoder_layers", type=int, default=2, help="Transformer Decoder 层数")
    parser.add_argument("--mlp_hidden", type=str, default="128,64", help="MLP 隐藏层维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 率")

    # 蛋白质编码参数
    parser.add_argument("--protein_method", type=str, default="esmc",
                        choices=["esmc", "acc", "onehot"],
                        help="蛋白质编码方法: esmc, acc, onehot")
    parser.add_argument("--protein_proj_method", type=str, default="conv",
                        choices=["conv", "mlp", "linear"],
                        help="ESMC蛋白质降维方法: conv, mlp, linear")

    # 分子编码器参数
    parser.add_argument("--gnn_type", type=str, default="gcn", choices=["gcn", "gat"],
                        help="GNN类型: gcn 或 gat")
    parser.add_argument("--gnn_layers", type=int, default=3, help="GNN层数")
    parser.add_argument("--gnn_hidden_dim", type=int, default=128, help="GNN隐藏层维度")

    # 优化参数
    parser.add_argument("--use_cosine", action="store_true", help="启用 cosine annealing 学习率调度")
    parser.add_argument("--early_stop_patience", type=int, default=30, help="早停耐心轮数（0=不启用）")

    # 融合方法参数
    parser.add_argument("--fusion_method", type=str, default="attention", choices=["attention", "concat"],
                        help="特征融合方法: attention（注意力机制）或 concat（直接拼接，用于消融实验）")

    # 官能团参数
    parser.add_argument("--no_fg_boost", action="store_false", dest="enable_fg_boost", default=True,
                        help="禁用官能团增强（对目标官能团不使用2.0倍权重，用于消融实验）")

    # 随机种子参数
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（默认42）")

    # 输出控制参数
    parser.add_argument("--quiet", action="store_true",
                        help="减少输出信息（仅显示关键信息）")

    args = parser.parse_args()

    # 设置随机种子（确保基本可复现性）
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

    device = torch.device(args.device)

    if not args.quiet:
        print(f"✓ 随机种子已设置为: {args.seed}")

    # 创建输出目录
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # 检查文件夹和文件
    folder = Path(args.folder)
    train_csv = folder / "train.csv"
    val_csv = folder / "val.csv"
    test_csv = folder / "test.csv"

    for f in [train_csv, val_csv, test_csv]:
        if not f.exists():
            print(f"错误: 文件不存在: {f}")
            return

    if not args.quiet:
        print(f"\n{'='*80}")
        print(f"MG2Act 蛋白端消融实验")
        print(f"蛋白质编码方法: {args.protein_method}")
        print(f"{'='*80}")
        print(f"数据文件夹: {args.folder}")
        print(f"设备: {device}")
        print(f"输出目录: {args.out}")
        print(f"架构配置:")
        print(f"  - 嵌入维度: {args.embed_dim}")
        print(f"  - 注意力头数: {args.attn_heads}")
        print(f"  - Decoder 层数: {args.decoder_layers}")
        print(f"  - 蛋白质编码方法: {args.protein_method}")
        if args.protein_method == "esmc":
            print(f"  - ESMC降维方法: {args.protein_proj_method}")
        print(f"  - 官能团增强: {'启用 (2.0倍权重)' if args.enable_fg_boost else '禁用 (1倍权重)'}")
        print(f"  - 融合方法: {args.fusion_method}")
        print(f"{'='*80}\n")

    # 训练
    best_val, best_val_mae, best_test_mae, total_time, threshold = train_fixed_split(
        train_csv, val_csv, test_csv, args, device
    )

    # 保存详细的训练结果（包含完整的回归指标）
    save_detailed_results_with_metrics(train_csv, val_csv, test_csv, args, device)


if __name__ == "__main__":
    main()
