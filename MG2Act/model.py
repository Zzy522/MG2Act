from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入拆分后的模块
from .encoder import EnhancedGNNMolecularEncoder, SequenceEncoder
from .attention_fusion import MultiScaleAttentionFusion, ConcatenationFusion
from .utils import LayerNorm, ConvPooler


class MG2ActModel(nn.Module):
    """
    MG2Act main model: Predicts PROTAC degradation activity (token-level attention version).

    Architecture features:
    1. Uses pretrained ESMC to encode protein sequences (returns token-level features)
    2. Uses GNN to encode SMILES molecules (returns node-level features)
    3. Uses Transformer Decoder cross-attention for token-level feature fusion
    4. Sequential interaction: molecular nodes -> E3 protein tokens -> target protein tokens
    5. Can focus on specific functional groups and amino acid residues relationships
    """
    
    def __init__(self,
                 device: torch.device = torch.device("cpu"),
                 embed_dim: int = 64,
                 attn_heads: int = 4,
                 decoder_layers: int = 2,
                 mlp_hidden: str = "128,64",
                 dropout: float = 0.1,
                 proj_method: str = "conv",  # "conv", "mlp", "linear"
                 # GNN分子编码器配置
                 gnn_type: str = "gcn",  # "gcn" 或 "gat"
                 gnn_layers: int = 3,  # GNN层数
                 gnn_hidden_dim: int = 128,  # GNN隐藏层维度
                 node_feat_dim: int = 44,  # 节点特征维度
                 edge_feat_dim: int = 10,  # 边特征维度
                 # 官能团注意力配置
                 use_multiscale_attention: bool = True,  # 是否使用多尺度注意力
                 custom_functional_groups: dict = None,  # 自定义官能团SMARTS模式
                 use_fg_only_attention: bool = True,  # 是否只使用官能团级别注意力（跳过原子级别）
                 enable_fg_boost: bool = True,  # 是否启用官能团增强（对目标官能团使用2倍权重）
                 fusion_method: str = "attention"):  # 融合方法: "attention" 或 "concat"
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim

        # Reduce output during model initialization (controlled by env var)
        import os
        verbose = os.environ.get('MG2ACT_VERBOSE', '0') == '1'
        self.use_multiscale_attention = use_multiscale_attention
        self.use_fg_only_attention = use_fg_only_attention

        # Default custom functional groups (user-provided)
        if custom_functional_groups is None:
            custom_functional_groups = {
                'cyclic_imide_1': 'C1(=O)CCCC(=O)N1',
                'cyclic_imide_2': 'C1(=O)CCNC(=O)N1',
                'cyclic_imide_3': 'N1C(=O)CCC1=O',
            }

        if verbose:
            print(f"[Model init] embed_dim={embed_dim}, attn_heads={attn_heads}, decoder_layers={decoder_layers}")
            print(f"[Model init] protein projection: {proj_method}")
            print(f"[Model init] GNN molecular encoder: {gnn_type}, layers={gnn_layers}")
            print(f"[Model init] multiscale attention: {use_multiscale_attention}")
            if custom_functional_groups:
                print(f"[Model init] custom functional groups: {len(custom_functional_groups)}")

        # 1. 蛋白序列编码器（ESMC，返回token级别特征）
        self.seq_encoder = SequenceEncoder(
            device=device,
            proj_method=proj_method,
            embed_dim=embed_dim,
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
            self.fusion = MultiScaleAttentionFusion(
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
                # nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))  # 输出单个分数
        
        self.prediction_head = nn.Sequential(*layers)
        
        # 确保所有模块在正确的设备上
        self.to(device)
        
        # Reduce output (print only in verbose mode)
        import os
        verbose = os.environ.get('MG2ACT_VERBOSE', '0') == '1'
        if verbose:
            print(f"[Model init] Complete, total params: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, 
                e3_seqs: List[str], 
                tgt_seqs: List[str], 
                smiles_list: List[str],
                return_attention: bool = False):
        """
        Token级别注意力前向传播

        参数:
            e3_seqs: E3连接酶序列列表 [B]
            tgt_seqs: 靶点蛋白序列列表 [B]
            smiles_list: SMILES分子列表 [B]
            return_attention: 是否返回注意力权重（仅返回第一个样本的注意力）

        返回:
            如果 return_attention=False:
                预测分数 [B]
            如果 return_attention=True:
                score: 预测分数 [B]
                attention_stage1: 阶段1原子注意力权重 [N_mol, L_e3]（第一个分子的，所有头平均后）
                attention_stage2: 阶段2原子注意力权重 [N_mol, L_tgt]（第一个分子的，所有头平均后）
                (如果use_multiscale_attention=True，还会返回:)
                fg_attention_stage1: 阶段1官能团注意力权重 [N_fg, L_e3]（第一个分子的，所有头平均后）
                fg_attention_stage2: 阶段2官能团注意力权重 [N_fg, L_tgt]（第一个分子的，所有头平均后）
                fg_info: 官能团信息列表 [List[List[Dict]]]（每个分子的官能团信息）
        """
        # 1. Encode protein sequences
        # Decide whether to return token-level features based on fusion method
        if self.fusion_method == "attention":
            # Attention method needs token-level features
            v_e3_tokens = self.seq_encoder(e3_seqs, return_token_level=True)   # List[[L_i, embed_dim]]
            v_tgt_tokens = self.seq_encoder(tgt_seqs, return_token_level=True) # List[[L_i, embed_dim]]
            v_e3_pooled = None
            v_tgt_pooled = None
        else:
            # Concat method: get pooled features directly, avoid computing token-level features
            v_e3_pooled = self.seq_encoder(e3_seqs, return_token_level=False)  # [B, embed_dim]
            v_tgt_pooled = self.seq_encoder(tgt_seqs, return_token_level=False)  # [B, embed_dim]
            v_e3_tokens = None
            v_tgt_tokens = None

        # 2. Encode molecules (return only functional group features)
        _, _, v_fg_features, fg_info = self.mol_encoder(smiles_list)

        # 3. 特征融合
        if self.fusion_method == "attention":
            fusion_output = self.fusion(
                v_mol_nodes=None,  # 不再使用原子级别特征
                mol_batch_indices=None,  # 不再使用原子索引
                v_fg_features=v_fg_features,
                v_e3_tokens=v_e3_tokens,
                v_tgt_tokens=v_tgt_tokens,
                return_attention=return_attention,
                fg_info=fg_info
            )
        else:
            # concat方法：直接使用池化特征
            fusion_output = self.fusion(
                v_mol_nodes=None,
                mol_batch_indices=None,
                v_fg_features=v_fg_features,
                v_e3_tokens=None,  # 不再使用token级别特征
                v_tgt_tokens=None,  # 不再使用token级别特征
                v_e3_pooled=v_e3_pooled,  # 直接传入池化特征
                v_tgt_pooled=v_tgt_pooled,  # 直接传入池化特征
                return_attention=return_attention,
                fg_info=fg_info
            )
        
        if return_attention:
            # 只返回官能团注意力
            h_final, h_stage1, fg_attention_stage1, fg_attention_stage2, fg_info = fusion_output
            attention_stage1 = None  # 不再使用原子注意力
            attention_stage2 = None  # 不再使用原子注意力
        else:
            h_final, h_stage1 = fusion_output
            attention_stage1 = None
            attention_stage2 = None
            fg_attention_stage1 = None
            fg_attention_stage2 = None
            fg_info = [[] for _ in range(len(smiles_list))]
            
        # h_final: Final fused features [B, embed_dim]
        # h_stage1: Molecular-E3 complex features [B, embed_dim]

        # 4. Concatenate features for prediction
        pred_input = torch.cat([h_final, h_stage1], dim=1)  # [B, 2*embed_dim]

        # 5. Predict degradation activity score
        score = self.prediction_head(pred_input).squeeze(-1)  # [B]

        if return_attention:
            # 只返回官能团注意力
            return score, fg_attention_stage1, fg_attention_stage2, fg_info
        else:
            return score
