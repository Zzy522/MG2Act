#!/usr/bin/env python3
"""
MG2Act Functional Group Attention Visualization Script

Features:
1. Load trained model
2. Input molecule SMILES, E3 sequence, target sequence
3. Perform attention calculation for both stages
4. Visualize attention weights as heatmaps
5. Output detailed functional group scoring explanations
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

# RDKit相关导入
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from MG2Actgithub_v1.model import MG2ActModel
from MG2Actgithub_v1.encoder import FunctionalGroupDetector


class AttentionVisualizer:
    """官能团注意力可视化器"""

    def __init__(self, model_path: str, device: str = "cuda:0"):
        """
        初始化可视化器

        参数:
            model_path: 模型文件路径
            device: 计算设备
        """
        self.device = torch.device(device)
        self.model_path = Path(model_path)

        # 加载模型配置
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get("config", {})

        # 获取训练时的官能团强化设置
        self.enable_fg_boost = config.get("enable_fg_boost", True)
        self.target_fg_boost = 2.0 if self.enable_fg_boost else 1.0

        # 重建模型
        self.model = self._load_model(checkpoint, config)
        self.model.eval()

        # 初始化官能团检测器（与模型使用相同的模式）
        self.fg_detector = FunctionalGroupDetector({
            'cyclic_imide_1': 'C1(=O)CCCC(=O)N1',
            'cyclic_imide_2': 'C1(=O)CCNC(=O)N1',
            'cyclic_imide_3': 'N1C(=O)CCC1=O',
        })

        print(f"✓ 模型加载完成: {model_path}")
        print(f"✓ 使用设备: {device}")
        print(f"✓ 训练时官能团强化: {'启用 (强化倍数: {:.1f})' if self.enable_fg_boost else '禁用'}".format(self.target_fg_boost))

    def _load_model(self, checkpoint: Dict, config: Dict) -> MG2ActModel:
        """加载模型"""
        # 默认配置
        default_config = {
            "embed_dim": 64,
            "attn_heads": 4,
            "decoder_layers": 2,
            "mlp_hidden": "128,64",
            "dropout": 0.1,
            "proj_method": "conv",
            "gnn_type": "gcn",
            "gnn_layers": 2,
            "gnn_hidden_dim": 64,
            "fusion_method": "attention",
            "enable_fg_boost": True  # 添加默认值
        }

        # 合并配置
        config = {**default_config, **config}

        # 创建模型
        model = MG2ActModel(
            device=self.device,
            embed_dim=config["embed_dim"],
            attn_heads=config["attn_heads"],
            decoder_layers=config["decoder_layers"],
            mlp_hidden=config["mlp_hidden"],
            dropout=config["dropout"],
            proj_method=config["proj_method"],
            gnn_type=config["gnn_type"],
            gnn_layers=config["gnn_layers"],
            gnn_hidden_dim=config["gnn_hidden_dim"],
            fusion_method=config["fusion_method"],
            enable_fg_boost=config["enable_fg_boost"]  # 传递参数
        )

        # 加载权重
        model.load_state_dict(checkpoint["model"])
        model.to(self.device)

        return model

    def analyze_attention(self, smiles: str, e3_seq: str, target_seq: str) -> Dict[str, Any]:
        """
        分析注意力权重

        参数:
            smiles: 分子SMILES
            e3_seq: E3连接酶序列
            target_seq: 靶点蛋白序列

        返回:
            包含预测结果和注意力权重的字典
        """
        with torch.no_grad():
            # 模型前向传播，获取注意力权重
            outputs = self.model(
                e3_seqs=[e3_seq],
                tgt_seqs=[target_seq],
                smiles_list=[smiles],
                return_attention=True
            )

            # 解析输出
            if len(outputs) >= 4:
                score, fg_attn_stage1, fg_attn_stage2, fg_info = outputs[:4]
            else:
                raise ValueError("模型没有返回足够的注意力信息")

            # 获取官能团信息
            fg_info_list = fg_info[0] if fg_info and len(fg_info) > 0 else []

            # 转换为numpy数组
            score_val = float(score.item())
            attn_s1 = fg_attn_stage1.cpu().numpy() if fg_attn_stage1 is not None else None
            attn_s2 = fg_attn_stage2.cpu().numpy() if fg_attn_stage2 is not None else None

            return {
                "smiles": smiles,
                "e3_seq": e3_seq,
                "target_seq": target_seq,
                "predicted_score": score_val,
                "functional_groups": fg_info_list,
                "attention_stage1": attn_s1,  # [num_fg, seq_len_e3]
                "attention_stage2": attn_s2,  # [num_fg, seq_len_tgt]
                "e3_seq_len": len(e3_seq),
                "target_seq_len": len(target_seq),
                "num_functional_groups": len(fg_info_list)
            }

    def create_heatmap(self, attention_matrix: np.ndarray,
                       row_labels: List[str],
                       col_labels: List[str],
                       title: str,
                       save_path: str = None) -> plt.Figure:
        """
        创建注意力权重热力图

        参数:
            attention_matrix: 注意力权重矩阵 [rows, cols]
            row_labels: 行标签（官能团名称）
            col_labels: 列标签（氨基酸序列）
            title: 图表标题
            save_path: 保存路径（可选）

        返回:
            matplotlib Figure对象
        """
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建图形
        fig, ax = plt.subplots(figsize=(max(10, len(col_labels) * 0.3), max(6, len(row_labels) * 0.5)))

        # 自定义颜色映射（从白色到蓝色）
        colors = [(1, 1, 1), (0.7, 0.9, 1), (0.4, 0.7, 1), (0, 0.4, 0.8), (0, 0, 0.6)]
        cmap = LinearSegmentedColormap.from_list("attention_cmap", colors, N=100)

        # 创建热力图
        sns.heatmap(attention_matrix,
                   annot=False,
                   cmap=cmap,
                   cbar=True,
                   cbar_kws={'label': '注意力权重'},
                   xticklabels=col_labels,
                   yticklabels=row_labels,
                   ax=ax,
                   linewidths=0.5,
                   linecolor='lightgray')

        # 设置标题和标签
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('蛋白质序列位置', fontsize=12)
        ax.set_ylabel('官能团', fontsize=12)

        # 调整x轴标签（每10个显示一个）
        if len(col_labels) > 20:
            step = max(1, len(col_labels) // 20)
            ax.set_xticks(range(0, len(col_labels), step))
            ax.set_xticklabels([col_labels[i] for i in range(0, len(col_labels), step)])

        # 旋转x轴标签
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 热力图已保存: {save_path}")

        return fig

    def generate_functional_group_report(self, analysis_result: Dict[str, Any]) -> str:
        """
        生成官能团注意力分析报告

        参数:
            analysis_result: analyze_attention的返回值

        返回:
            详细的分析报告字符串
        """
        fg_info = analysis_result["functional_groups"]
        attn_s1 = analysis_result["attention_stage1"]
        attn_s2 = analysis_result["attention_stage2"]

        report = []
        report.append("=" * 80)
        report.append("🧪 Functional Group Attention Analysis Report")
        report.append("=" * 80)
        report.append(f"Molecule SMILES: {analysis_result['smiles']}")
        report.append(f"Predicted Score: {analysis_result['predicted_score']:.4f}")
        report.append(f"Functional Groups Detected: {len(fg_info)}")
        report.append("")

        # 列出所有检测到的官能团及其原子序号
        if fg_info:
            report.append("📋 Detected Functional Groups:")
            for i, fg in enumerate(fg_info, 1):
                atom_indices = fg.get('atoms', [])
                atom_indices_str = ', '.join(map(str, atom_indices))
                report.append(f"   {i}. {fg['name']} - 原子序号: [{atom_indices_str}]")
            report.append("")

        # Stage 1 Analysis: Functional Groups vs E3 Ligase
        report.append("📊 Stage 1 Analysis: Molecular Functional Groups → E3 Ligase")
        report.append("-" * 50)

        if attn_s1 is not None and len(fg_info) > 0:
            # 计算每个官能团对E3蛋白的平均注意力
            fg_attention_to_e3 = attn_s1.mean(axis=1)  # [num_fg]

            # 按注意力权重排序
            sorted_indices = np.argsort(fg_attention_to_e3)[::-1]

            for rank, fg_idx in enumerate(sorted_indices, 1):
                fg = fg_info[fg_idx]
                attention_score = fg_attention_to_e3[fg_idx]
                print(f"Debug - fg_idx: {fg_idx}, attn_s1.shape: {attn_s1.shape}")
                print(f"Debug - len(e3_seq): {len(analysis_result['e3_seq'])}, attn_s1.shape[1]: {attn_s1.shape[1]}")
                print(f"Debug - attn_s1[fg_idx] length: {len(attn_s1[fg_idx])}")
                print(f"Debug - max position from argsort: {np.argsort(attn_s1[fg_idx])[::-1][:3]}")
                # 找出该官能团最关注的E3位置
                e3_positions = np.argsort(attn_s1[fg_idx])[::-1][:3]  # top 3
                
                # 确保位置不超出注意力矩阵的范围
                max_valid_pos = attn_s1.shape[1] - 1
                e3_positions = [pos for pos in e3_positions if pos <= max_valid_pos][:3]
                
                # 只使用注意力矩阵对应长度的序列部分，避免索引越界
                # e3_seq_for_attention = analysis_result['e3_seq'][:attn_s1.shape[1]]
                # top_e3_residues = [e3_seq_for_attention[pos] for pos in e3_positions]

                # 显示官能团包含的原子序号
                atom_indices = fg.get('atoms', [])
                atom_indices_str = ', '.join(map(str, atom_indices))

                report.append(f"{rank}. {fg['name']} (原子数: {fg['num_atoms']}, 原子序号: [{atom_indices_str}])")
                report.append(f"   Attention Score: {attention_score:.4f}")
                # report.append(f"   Top E3 Residues: {', '.join(top_e3_residues)}")
                report.append("")
        else:
            report.append("⚠️ Unable to get Stage 1 attention weights")
            report.append("")

        # Stage 2 Analysis: Functional Groups vs Target Protein
        report.append("📊 Stage 2 Analysis: Molecule-E3 Complex → Target Protein")
        report.append("-" * 50)

        if attn_s2 is not None and len(fg_info) > 0:
            # 计算每个官能团对靶点蛋白的平均注意力
            fg_attention_to_tgt = attn_s2.mean(axis=1)  # [num_fg]

            # 按注意力权重排序
            sorted_indices = np.argsort(fg_attention_to_tgt)[::-1]

            for rank, fg_idx in enumerate(sorted_indices, 1):
                fg = fg_info[fg_idx]
                attention_score = fg_attention_to_tgt[fg_idx]

                # 找出该官能团最关注的靶点位置
                tgt_positions = np.argsort(attn_s2[fg_idx])[::-1][:3]  # top 3
                # 过滤掉超出序列长度的位置
                tgt_seq_len = len(analysis_result['target_seq'])
                valid_positions = [pos for pos in tgt_positions if pos < tgt_seq_len][:3]
                
                if valid_positions:
                    top_tgt_residues = [analysis_result['target_seq'][pos] for pos in valid_positions]
                else:
                    top_tgt_residues = ['N/A']

                # 显示官能团包含的原子序号
                atom_indices = fg.get('atoms', [])
                atom_indices_str = ', '.join(map(str, atom_indices))

                report.append(f"{rank}. {fg['name']} (原子数: {fg['num_atoms']}, 原子序号: [{atom_indices_str}])")
                report.append(f"   Attention Score: {attention_score:.4f}")
                report.append(f"   Top Target Residues: {', '.join(top_tgt_residues)}")
                report.append("")
        else:
            report.append("⚠️ Unable to get Stage 2 attention weights")
            report.append("")

        # Key Insights
        report.append("🔍 Key Insights")
        report.append("-" * 50)

        if attn_s1 is not None and attn_s2 is not None and len(fg_info) > 0:
            # 分析哪些官能团在两个阶段都很重要
            fg_s1_scores = attn_s1.mean(axis=1)
            fg_s2_scores = attn_s2.mean(axis=1)

            # 计算两个阶段的相关性
            correlation = np.corrcoef(fg_s1_scores, fg_s2_scores)[0, 1]
            report.append(".4f")
            # 找出在两个阶段都很重要的官能团
            combined_scores = (fg_s1_scores + fg_s2_scores) / 2
            top_combined_idx = np.argmax(combined_scores)
            top_fg = fg_info[top_combined_idx]

            # 显示原子序号
            atom_indices = top_fg.get('atoms', [])
            atom_indices_str = ', '.join(map(str, atom_indices))

            report.append(f"⭐ Most Important Functional Group: {top_fg['name']} (原子序号: [{atom_indices_str}], Avg Attention: {(combined_scores[top_combined_idx]):.4f})")

            # Check for cyclic imides (key functional groups in molecular glues)
            cyclic_imides = [fg for fg in fg_info if 'cyclic_imide' in fg['name']]
            if cyclic_imides:
                ci_scores = []
                for ci in cyclic_imides:
                    idx = fg_info.index(ci)
                    atom_indices = ci.get('atoms', [])
                    atom_indices_str = ', '.join(map(str, atom_indices))
                    ci_scores.append((ci['name'], (fg_s1_scores[idx] + fg_s2_scores[idx]) / 2, atom_indices_str))

                best_ci = max(ci_scores, key=lambda x: x[1])
                report.append(f"🧬 Best Cyclic Imide: {best_ci[0]} (原子序号: [{best_ci[2]}], Attention: {best_ci[1]:.4f})")
            else:
                report.append("⚠️ No cyclic imide functional groups detected")

        report.append("=" * 80)

        return "\n".join(report)

    def visualize_attention(self, analysis_result: Dict[str, Any],
                          output_dir: str = "attention_visualization") -> Dict[str, str]:
        """
        生成完整的注意力可视化

        参数:
            analysis_result: analyze_attention的返回值
            output_dir: 输出目录

        返回:
            保存的文件路径字典
        """
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        saved_files = {}

        # 1. 生成文字报告
        report = self.generate_functional_group_report(analysis_result)
        report_file = output_path / "attention_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        saved_files['report'] = str(report_file)
        print(f"✓ Analysis report saved: {report_file}")

        # 2. Stage 1 visualization: Functional Groups vs E3 Ligase
        if analysis_result["attention_stage1"] is not None:
            attn_s1 = analysis_result["attention_stage1"]

            # Prepare labels
            fg_names = [fg['name'] for fg in analysis_result["functional_groups"]]
            e3_residues = list(analysis_result["e3_seq"])

            # 创建热力图
            fig1 = self.create_heatmap(
                attn_s1,
                row_labels=fg_names,
                col_labels=e3_residues,
                title="Stage 1: Molecular Functional Groups → E3 Ligase Attention Weights",
                save_path=output_path / "stage1_attention_heatmap.png"
            )
            saved_files['stage1_heatmap'] = str(output_path / "stage1_attention_heatmap.png")
            plt.close(fig1)

        # 3. Stage 2 visualization: Functional Groups vs Target Protein
        if analysis_result["attention_stage2"] is not None:
            attn_s2 = analysis_result["attention_stage2"]

            # For long target sequences, show only first 100 amino acids
            seq_len = len(analysis_result["target_seq"])
            display_len = min(100, seq_len)

            attn_s2_display = attn_s2[:, :display_len]
            tgt_residues = list(analysis_result["target_seq"][:display_len])

            # 准备标签
            fg_names = [fg['name'] for fg in analysis_result["functional_groups"]]

            # 创建热力图
            fig2 = self.create_heatmap(
                attn_s2_display,
                row_labels=fg_names,
                col_labels=tgt_residues,
                title=f"Stage 2: Molecule-E3 Complex → Target Protein Attention Weights (First {display_len} residues)",
                save_path=output_path / "stage2_attention_heatmap.png"
            )
            saved_files['stage2_heatmap'] = str(output_path / "stage2_attention_heatmap.png")
            plt.close(fig2)

        # 4. Calculate global maximum attention score for unified normalization
        attn_s1 = analysis_result["attention_stage1"]
        attn_s2 = analysis_result["attention_stage2"]

        # Calculate global maximum across both stages
        global_max = 0.0
        if attn_s1 is not None:
            global_max = max(global_max, attn_s1.max())
        if attn_s2 is not None:
            global_max = max(global_max, attn_s2.max())

        # Create 2D molecule attention visualization (separate for each stage)
        mol_vis_stage1 = self.create_molecule_attention_visualization(
            analysis_result, output_path, stage="stage1"
        )
        mol_vis_stage2 = self.create_molecule_attention_visualization(
            analysis_result, output_path, stage="stage2"
        )
        mol_vis_combined = self.create_molecule_attention_visualization(
            analysis_result, output_path, stage="combined"
        )

        if mol_vis_stage1:
            saved_files['molecule_2d_stage1'] = mol_vis_stage1
        if mol_vis_stage2:
            saved_files['molecule_2d_stage2'] = mol_vis_stage2
        if mol_vis_combined:
            saved_files['molecule_2d_combined'] = mol_vis_combined

        # 5. 保存详细数据为JSON
        # 将numpy数组转换为列表以支持JSON序列化
        json_result = {}
        for key, value in analysis_result.items():
            if hasattr(value, 'tolist'):  # numpy数组
                json_result[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and hasattr(value[0], 'tolist'):
                # 处理包含numpy数组的列表
                json_result[key] = [item.tolist() if hasattr(item, 'tolist') else item for item in value]
            else:
                json_result[key] = value

        json_data = {
            "analysis_result": json_result,
            "report_summary": report.split('\n')[:10]  # 前10行摘要
        }
        json_file = output_path / "attention_data.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        saved_files['json_data'] = str(json_file)

        print(f"✓ Detailed data saved: {json_file}")
        print("\n" + "="*60)
        print("🎉 Visualization Complete!")
        print("="*60)
        print(f"📁 Output Directory: {output_path.absolute()}")
        print(f"📄 Analysis Report: {saved_files['report']}")

        if 'stage1_heatmap' in saved_files:
            print(f"🔥 Stage 1 Heatmap: {saved_files['stage1_heatmap']}")
        if 'stage2_heatmap' in saved_files:
            print(f"🔥 Stage 2 Heatmap: {saved_files['stage2_heatmap']}")
        if 'molecule_2d_stage1' in saved_files:
            print(f"🧪 2D Molecule Attention (Stage 1): {saved_files['molecule_2d_stage1']}")
        if 'molecule_2d_stage2' in saved_files:
            print(f"🧪 2D Molecule Attention (Stage 2): {saved_files['molecule_2d_stage2']}")
        if 'molecule_2d_combined' in saved_files:
            print(f"🧪 2D Molecule Attention (Combined): {saved_files['molecule_2d_combined']}")

        print(f"📊 Detailed Data: {saved_files['json_data']}")
        print("="*60)

        return saved_files

    def create_molecule_attention_visualization(self, analysis_result: Dict[str, Any],
                                               output_path: Path, stage: str = "combined",
                                               global_max_score: float = None) -> str:
        """
        创建分子2D结构图，显示原子级别的注意力信息

        参数:
            analysis_result: analyze_attention的返回值
            output_path: 输出目录路径

        返回:
            保存的图像文件路径
        """
        smiles = analysis_result["smiles"]
        fg_info = analysis_result["functional_groups"]
        attn_s1 = analysis_result["attention_stage1"]  # [num_fg, seq_len_e3]
        attn_s2 = analysis_result["attention_stage2"]  # [num_fg, seq_len_tgt]

        # 1. 根据阶段选择注意力权重
        if stage == "stage1":
            attn_to_use = attn_s1
            stage_name = "阶段1"
        elif stage == "stage2":
            attn_to_use = attn_s2
            stage_name = "阶段2"
        else:  # combined
            # 使用两个阶段的平均值（分别计算每个阶段的平均注意力）
            if attn_s1 is not None and attn_s2 is not None:
                # 计算每个阶段的平均注意力分数，然后平均
                stage1_avg = attn_s1.mean(axis=1, keepdims=True)  # [num_fg, 1]
                stage2_avg = attn_s2.mean(axis=1, keepdims=True)  # [num_fg, 1]
                combined_avg = (stage1_avg + stage2_avg) / 2  # [num_fg, 1]

                # 将平均分数广播到所有位置（为了可视化的一致性）
                if attn_s1.shape[1] <= attn_s2.shape[1]:
                    # 使用更短的序列长度
                    attn_to_use = combined_avg * np.ones_like(attn_s1)
                else:
                    attn_to_use = combined_avg * np.ones_like(attn_s2)
            elif attn_s1 is not None:
                attn_to_use = attn_s1
            elif attn_s2 is not None:
                attn_to_use = attn_s2
            else:
                attn_to_use = None
            stage_name = "综合"

        # 2. 将官能团注意力映射到原子级别
        atom_attention_scores = self._map_fg_attention_to_atoms(smiles, fg_info, attn_to_use, global_max_score)

        if atom_attention_scores is None:
            print("⚠️ 无法创建分子注意力可视化")
            return None

        # 2. 创建带有注意力着色的分子图
        mol_img_path = output_path / f"molecule_attention_2d_{stage}.png"
        self._draw_molecule_with_attention(smiles, atom_attention_scores, str(mol_img_path), stage_name, stage)

        print(f"✓ {stage_name} molecule attention visualization saved: {mol_img_path}")
        return str(mol_img_path)

        print(f"✓ 分子注意力可视化已保存: {mol_img_path}")
        return str(mol_img_path)

    def _map_fg_attention_to_atoms(self, smiles: str, fg_info: List[Dict],
                                  attn_matrix: np.ndarray, unused=None) -> np.ndarray:
        """
        将官能团级别的注意力分数映射到原子级别

        参数:
            smiles: 分子SMILES
            fg_info: 官能团信息列表
            attn_s1: 阶段1注意力权重 [num_fg, seq_len_e3]
            attn_s2: 阶段2注意力权重 [num_fg, seq_len_tgt]

        返回:
            atom_scores: 每个原子的注意力分数 [num_atoms]
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            num_atoms = mol.GetNumAtoms()
            atom_scores = np.zeros(num_atoms)

            # 计算每个官能团的注意力分数
            if attn_matrix is not None:
                fg_scores = attn_matrix.mean(axis=1)  # 对序列维度取平均
            else:
                return None

            # 将官能团分数分配给其包含的原子（取最大值而非累加）
            atom_to_fg_scores = {}  # 记录每个原子所属的所有官能团的注意力分数

            for fg_idx, fg in enumerate(fg_info):
                fg_score = fg_scores[fg_idx]
                atom_indices = fg.get('atoms', [])

                # 为该官能团的所有原子记录注意力分数
                for atom_idx in atom_indices:
                    if atom_idx < num_atoms:
                        if atom_idx not in atom_to_fg_scores:
                            atom_to_fg_scores[atom_idx] = []
                        atom_to_fg_scores[atom_idx].append(fg_score)

            # 对于每个原子，取其所属官能团中注意力最大的分数
            for atom_idx in range(num_atoms):
                if atom_idx in atom_to_fg_scores and len(atom_to_fg_scores[atom_idx]) > 0:
                    # 取最大注意力分数
                    atom_scores[atom_idx] = max(atom_to_fg_scores[atom_idx])
                else:
                    atom_scores[atom_idx] = 0.0

            # 归一化到[0,1]区间（每个阶段独立归一化）
            if atom_scores.max() > 0:
                atom_scores = atom_scores / atom_scores.max()

            return atom_scores

        except Exception as e:
            print(f"映射原子注意力时出错: {e}")
            return None

    def _draw_molecule_with_attention(self, smiles: str, atom_scores: np.ndarray, save_path: str, stage_name: str = "Combined", stage: str = "combined"):
        """
        绘制带有注意力着色的分子2D结构图，并标注原子序号

        参数:
            smiles: 分子SMILES
            atom_scores: 每个原子的注意力分数 [num_atoms]
            save_path: 保存路径
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return

            # 计算颜色映射（统一使用蓝色系：深蓝=高注意力，浅蓝=低注意力）
            colors = {}
            for atom_idx, score in enumerate(atom_scores):
                if score > 0:
                    # 统一的蓝色系：高注意力使用深蓝色，低注意力使用浅蓝色
                    r = max(0.0, 1.0 - score * 0.8)  # 红色分量：高注意力时接近0
                    g = max(0.0, 1.0 - score * 0.6)  # 绿色分量：中等减少
                    b = min(1.0, 0.4 + score * 0.6)   # 蓝色分量：高注意力时接近1

                    colors[atom_idx] = (r, g, b)
                else:
                    # 无注意力原子使用浅灰色
                    colors[atom_idx] = (0.9, 0.9, 0.9)

            # 创建2D坐标
            rdDepictor.Compute2DCoords(mol)

            # 使用RDKit的2D绘图器
            drawer = rdMolDraw2D.MolDraw2DCairo(800, 600)
            drawer.drawOptions().useBWAtomPalette()

            # 设置原子颜色
            for atom_idx in colors:
                color = colors[atom_idx]
                drawer.drawOptions().setAtomPalette({atom_idx: color})

            # 启用原子序号标注
            drawer.drawOptions().addAtomIndices = True
            drawer.drawOptions().addBondIndices = False  # 不显示键序号

            # 绘制分子
            drawer.DrawMolecule(mol, highlightAtoms=list(colors.keys()),
                              highlightAtomColors=colors)
            drawer.FinishDrawing()

            # 保存图像
            drawer.WriteDrawingText(save_path)

            # 创建颜色图例
            self._create_color_legend(save_path.replace('.png', '_legend.png'), stage)

        except Exception as e:
            print(f"绘制分子注意力图时出错: {e}")
            # 备用方案：创建简单的文本表示
            self._create_text_attention_visualization(smiles, atom_scores, save_path)

    def _create_text_attention_visualization(self, smiles: str, atom_scores: np.ndarray, save_path: str):
        """
        创建文本形式的注意力可视化（当RDKit绘图失败时的备用方案）

        参数:
            smiles: 分子SMILES
            atom_scores: 原子注意力分数
            save_path: 保存路径
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return

            # Create text report
            lines = []
            lines.append("🧪 Molecule Attention Visualization (Text Mode)")
            lines.append("=" * 50)
            lines.append(f"SMILES: {smiles}")
            lines.append("")

            # Display attention scores for each atom
            lines.append("Atom Attention Scores:")
            lines.append("-" * 30)

            for atom_idx in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(atom_idx)
                symbol = atom.GetSymbol()
                score = atom_scores[atom_idx] if atom_idx < len(atom_scores) else 0.0

                # 根据分数确定重要性标记
                if score > 0.7:
                    marker = "🔴"  # 高注意力
                elif score > 0.4:
                    marker = "🟡"  # 中注意力
                elif score > 0.1:
                    marker = "🟢"  # 低注意力
                else:
                    marker = "⚪"  # 无注意力

                lines.append("6s")

            lines.append("")
            lines.append("Legend:")
            lines.append("🔴 High Attention (score > 0.7)")
            lines.append("🟡 Medium Attention (0.4 < score ≤ 0.7)")
            lines.append("🟢 Low Attention (0.1 < score ≤ 0.4)")
            lines.append("⚪ No Attention (score ≤ 0.1)")

            # 保存为文本文件
            text_path = save_path.replace('.png', '.txt')
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))

            print(f"✓ 文本注意力可视化已保存: {text_path}")

        except Exception as e:
            print(f"创建文本可视化时出错: {e}")

    def _create_color_legend(self, legend_path: str, stage: str):
        """
        创建颜色图例

        参数:
            legend_path: 图例保存路径
            stage: 阶段名称
        """
        try:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

            # Unified blue color scheme for all stages
            colors = [(0.6, 0.4, 0.4), (0.3, 0.5, 0.7), (0.2, 0.4, 1.0)]
            labels = ["Low Attention", "Medium Attention", "High Attention"]

            if stage == "stage1":
                title = "Stage 1: Molecule→E3 Ligase\nBlue intensity indicates attention strength"
            elif stage == "stage2":
                title = "Stage 2: Molecule→Target Protein\nBlue intensity indicates attention strength"
            else:
                title = "Combined Stage\nBlue intensity indicates average attention"

            # 绘制颜色块
            for i, (color, label) in enumerate(zip(colors, labels)):
                y_pos = 0.8 - i * 0.2
                ax.add_patch(plt.Rectangle((0.1, y_pos), 0.2, 0.15, facecolor=color))
                ax.text(0.35, y_pos + 0.075, label, fontsize=10, verticalalignment='center')

            # 添加标题
            ax.text(0.5, 0.95, title, fontsize=12, ha='center', va='top', fontweight='bold')

            plt.savefig(legend_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            print(f"✓ 颜色图例已保存: {legend_path}")

        except Exception as e:
            print(f"创建颜色图例时出错: {e}")


def main():
    parser = argparse.ArgumentParser(description="MG2Act 官能团注意力可视化工具")
    parser.add_argument("--model_path", type=str, required=True,
                       help="训练好的模型文件路径")
    parser.add_argument("--smiles", type=str, required=True,
                       help="分子SMILES字符串")
    parser.add_argument("--e3_seq", type=str, required=True,
                       help="E3连接酶序列")
    parser.add_argument("--target_seq", type=str, required=True,
                       help="靶点蛋白序列")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="计算设备 (cuda:0 或 cpu)")
    parser.add_argument("--output_dir", type=str, default="attention_visualization",
                       help="输出目录")

    args = parser.parse_args()

    # 检查文件是否存在
    if not Path(args.model_path).exists():
        print(f"错误：模型文件不存在: {args.model_path}")
        return

    # 创建输出目录
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)

    try:
        # 初始化可视化器
        print("正在加载模型...")
        visualizer = AttentionVisualizer(args.model_path, args.device)

        # 分析注意力
        print("正在分析注意力权重...")
        analysis_result = visualizer.analyze_attention(
            args.smiles, args.e3_seq, args.target_seq
        )

        # 生成可视化
        print("正在生成可视化...")
        saved_files = visualizer.visualize_attention(analysis_result, args.output_dir)

        # 打印分析报告
        print("\n" + "="*80)
        print("📋 分析摘要")
        print("="*80)
        print(".4f")
        print(f"检测到官能团: {analysis_result['num_functional_groups']} 个")

        if analysis_result['functional_groups']:
            fg_names = [fg['name'] for fg in analysis_result['functional_groups']]
            print(f"官能团类型: {', '.join(fg_names)}")

        print("="*80)

    except Exception as e:
        print(f"错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
"""
python MG2Actgithub_v1/attention_visualizer.py --model_path model_new/test1nofg/mg2act_best.pt --smiles "C(NC(=O)NC1C=CC(CNC(=O)CN2C3N=CC=CC=3C(C)=C2)=CC=1)1C=C2CN(C3C(=O)NC(=O)CC3)C(=O)C2=CC=1" --e3_seq "CTSLCCKQCQETEITTKNEIFSLSLCGPMAAYVNPHGYVHETLTVYKACNLNLIGRPSTEHSWFPGYAWTVAQCKICASHIGWKFTATKKDMSPQKFWGLTRSALLPTIPDTEDEISPDKVILCL" --target_seq "MDADEGQDMSQVSGKESPPVSDTPDEGDEPMPIPEDLSTTSGGQQSSKSDRVVASNVKVETQSDEENGRACEMNGEECAEDLRMLDASGEKMNGSHRDQGSSALSGVGGIRLPNGKLKCDICGIICIGPNVLMVHKRSHTGERPFQCNQCGASFTQKGNLLRHIKLHSGEKPFKCHLCNYACRRRDALTGHLRTHSVGKPHKCGYCGRSYKQRSSLEEHKERCHNYLESMGLPGTLYPVIKEETNHSEMAEDLCKIGSERSLVLDRLASNVAKRKSSMPQKFLGDKGLSDTPYDSSASYEKENEMMKSHVMDQAINNAINYLGAESLRPLVQTPPGGSEVVPVISPMYQLHKPLAEGTPRSNHSAQDSAVENLLLLSKAKLVPSEREASPSNSCQDSTDTESNNEEQRSGLIYLTNHIAPHARNGLSLKEEHRAYDLLRAASENSQDALRVVSTSGEQMKVYKCEHCRVLFLDHVMYTIHMGCHGFRDPFECNMCGYHSQDRYEFSSHITRGEHRFHMS" --device cuda:0 --output_dir test_unified_normalization2
"""
