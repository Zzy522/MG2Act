"""
消融实验MG2Act模型
支持不同的蛋白质编码方法：ESMC, ACC, OneHot
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入现有模块
from ..encoder import EnhancedGNNMolecularEncoder
from ..attention_fusion import MultiScaleAttentionFusion, ConcatenationFusion
from .protein_encoders import AblationProteinEncoder
from .pooled_attention_fusion import PooledAttentionFusion


class MG2ActAblationModel(nn.Module):
    """
    MG2Act消融实验模型：支持不同的蛋白质编码方法

    架构特点：
    1. 可选择不同的蛋白质编码器（ESMC, ACC, OneHot）
    2. 使用 GNN 编码 SMILES 分子（返回节点级别特征）
    3. 使用 Transformer Decoder 的 cross-attention 进行特征融合
    4. 顺序交互：分子节点 -> E3蛋白 -> 靶点蛋白
    """

    def __init__(self,
                 device: torch.device = torch.device("cpu"),
                 embed_dim: int = 64,
                 attn_heads: int = 4,
                 decoder_layers: int = 2,
                 mlp_hidden: str = "128,64",
                 dropout: float = 0.1,
                 # 蛋白质编码器配置
                 protein_method: str = "esmc",  # "esmc", "acc", "onehot"
                 protein_proj_method: str = "conv",  # ESMC专用
                 # GNN分子编码器配置
                 gnn_type: str = "gcn",
                 gnn_layers: int = 3,
                 gnn_hidden_dim: int = 128,
                 node_feat_dim: int = 44,
                 edge_feat_dim: int = 10,
                 # 官能团注意力配置
                 use_multiscale_attention: bool = True,
                 custom_functional_groups: dict = None,
                 enable_fg_boost: bool = True,
                 fusion_method: str = "attention"):  # "attention" 或 "concat"
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.protein_method = protein_method.lower()

        # 减少模型初始化时的输出
        import os
        verbose = os.environ.get('MG2ACT_VERBOSE', '0') == '1'

        # 默认自定义官能团
        if custom_functional_groups is None:
            custom_functional_groups = {
                'cyclic_imide_1': 'C1(=O)CCCC(=O)N1',
                'cyclic_imide_2': 'C1(=O)CCNC(=O)N1',
                'cyclic_imide_3': 'N1C(=O)CCC1=O',
            }

        if verbose:
            print(f"[消融模型初始化] 蛋白质编码方法: {protein_method}")
            print(f"[消融模型初始化] 嵌入维度={embed_dim}, 注意力头数={attn_heads}, Decoder层数={decoder_layers}")

        # 1. 蛋白质序列编码器（支持多种方法）
        self.seq_encoder = AblationProteinEncoder(
            method=protein_method,
            output_dim=embed_dim,
            device=device,
            proj_method=protein_proj_method,
            dropout=dropout
        )

        # 2. 分子编码器（只返回官能团特征）
        self.mol_encoder = EnhancedGNNMolecularEncoder(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_layers,
            output_dim=embed_dim,
            gnn_type=gnn_type,
            dropout=dropout,
            custom_functional_groups=custom_functional_groups
        ).to(device)

        # 3. 特征融合模块
        self.fusion_method = fusion_method
        if fusion_method == "attention":
            if protein_method == "esmc":
                # ESMC支持token级别特征，使用完整的MultiScaleAttentionFusion
                self.fusion = MultiScaleAttentionFusion(
                    embed_dim=embed_dim,
                    num_heads=attn_heads,
                    num_layers=decoder_layers,
                    dropout=dropout,
                    enable_fg_boost=enable_fg_boost
                )
            else:
                # ACC/OneHot只支持池化特征，使用PooledAttentionFusion
                self.fusion = PooledAttentionFusion(
                    embed_dim=embed_dim,
                    num_heads=attn_heads,
                    num_layers=decoder_layers,
                    dropout=dropout,
                    enable_fg_boost=enable_fg_boost
                )
        elif fusion_method == "concat":
            self.fusion = ConcatenationFusion(
                embed_dim=embed_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(f"不支持的融合方法: {fusion_method}")

        # 4. 预测头
        hidden_dims = [int(x) for x in mlp_hidden.split(',')]

        # 输入：最终融合特征 + 阶段1特征
        input_dim = embed_dim * 2  # h_final + h_stage1

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))  # 输出单个分数

        self.prediction_head = nn.Sequential(*layers)

        # 确保所有模块在正确的设备上
        self.to(device)

        # 减少输出
        if verbose:
            print(f"[消融模型初始化] 完成，总参数量: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self,
                e3_seqs: List[str],
                tgt_seqs: List[str],
                smiles_list: List[str],
                return_attention: bool = False):
        """
        前向传播

        参数:
            e3_seqs: E3连接酶序列列表 [B]
            tgt_seqs: 靶点蛋白序列列表 [B]
            smiles_list: SMILES分子列表 [B]
            return_attention: 是否返回注意力权重

        返回:
            如果 return_attention=False:
                预测分数 [B]
            如果 return_attention=True:
                score, fg_attention_stage1, fg_attention_stage2, fg_info
        """
        # 1. 编码蛋白质序列
        if self.protein_method == "esmc" and self.fusion_method == "attention":
            # ESMC + 注意力融合：需要token级别特征
            v_e3_tokens = self.seq_encoder(e3_seqs, return_token_level=True)
            v_tgt_tokens = self.seq_encoder(tgt_seqs, return_token_level=True)
            v_e3_pooled = None
            v_tgt_pooled = None
        else:
            # 其他情况：使用池化特征
            v_e3_pooled = self.seq_encoder(e3_seqs, return_token_level=False)
            v_tgt_pooled = self.seq_encoder(tgt_seqs, return_token_level=False)
            v_e3_tokens = None
            v_tgt_tokens = None

        # 2. 编码分子（只返回官能团特征）
        _, _, v_fg_features, fg_info = self.mol_encoder(smiles_list)

        # 3. 特征融合
        if self.protein_method == "esmc" and self.fusion_method == "attention":
            # ESMC + MultiScaleAttentionFusion（token级别注意力）
            fusion_output = self.fusion(
                v_mol_nodes=None,
                mol_batch_indices=None,
                v_fg_features=v_fg_features,
                v_e3_tokens=v_e3_tokens,
                v_tgt_tokens=v_tgt_tokens,
                return_attention=return_attention,
                fg_info=fg_info
            )
        elif self.protein_method != "esmc" and self.fusion_method == "attention":
            # ACC/OneHot + PooledAttentionFusion（池化特征注意力）
            fusion_output = self.fusion(
                v_mol_nodes=None,
                mol_batch_indices=None,
                v_fg_features=v_fg_features,
                v_e3_tokens=None,
                v_tgt_tokens=None,
                v_e3_pooled=v_e3_pooled,
                v_tgt_pooled=v_tgt_pooled,
                return_attention=return_attention,
                fg_info=fg_info
            )
        else:
            # ConcatenationFusion（直接拼接，用于消融实验）
            fusion_output = self.fusion(
                v_mol_nodes=None,
                mol_batch_indices=None,
                v_fg_features=v_fg_features,
                v_e3_tokens=None,
                v_tgt_tokens=None,
                v_e3_pooled=v_e3_pooled,
                v_tgt_pooled=v_tgt_pooled,
                return_attention=return_attention,
                fg_info=fg_info
            )

        if return_attention:
            h_final, h_stage1, fg_attention_stage1, fg_attention_stage2, fg_info = fusion_output
        else:
            h_final, h_stage1 = fusion_output
            fg_attention_stage1 = None
            fg_attention_stage2 = None
            fg_info = [[] for _ in range(len(smiles_list))]

        # 4. 拼接特征进行预测
        pred_input = torch.cat([h_final, h_stage1], dim=1)  # [B, 2*embed_dim]

        # 5. 预测降解活性得分
        score = self.prediction_head(pred_input).squeeze(-1)  # [B]

        if return_attention:
            return score, fg_attention_stage1, fg_attention_stage2, fg_info
        else:
            return score
