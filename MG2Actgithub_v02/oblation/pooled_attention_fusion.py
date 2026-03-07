"""
池化特征注意力融合模块
用于ACC/OneHot等不支持token级别特征的编码方法

这个模块使用池化后的蛋白质特征（而不是token级别特征）进行注意力融合
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PooledAttentionFusion(nn.Module):
    """
    基于池化特征的注意力融合
    
    适用于ACC/OneHot等只能输出池化特征的编码方法
    使用简化的注意力机制：官能团特征 attend to 池化的蛋白质特征
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, num_layers: int = 2, 
                 dropout: float = 0.1, enable_fg_boost: bool = True, fg_boost_factor: float = 2.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.target_fg_names = {'cyclic_imide_1', 'cyclic_imide_2', 'cyclic_imide_3'}
        self.target_fg_boost = fg_boost_factor if enable_fg_boost else 1.0

        # 阶段1: 官能团 attend to E3特征（池化）
        # 将E3池化特征扩展为"伪token"以便使用多头注意力
        self.e3_expander = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.LayerNorm(embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.stage1_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.stage1_norm = nn.LayerNorm(embed_dim)
        self.stage1_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

        # 阶段2: 官能团-E3复合物 attend to 靶点特征（池化）
        self.tgt_expander = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.LayerNorm(embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.stage2_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.stage2_norm = nn.LayerNorm(embed_dim)
        self.stage2_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def _pad_fg_sequences(self, fg_sequences: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """将变长官能团序列padding"""
        if len(fg_sequences) == 0 or all(len(fgs) == 0 for fgs in fg_sequences):
            device = torch.device("cpu")
            if len(fg_sequences) > 0:
                for fgs in fg_sequences:
                    if len(fgs) > 0:
                        device = fgs[0].device
                        break
            return torch.empty(0, 0, self.embed_dim, device=device), torch.empty(0, dtype=torch.long, device=device)

        device = None
        for fgs in fg_sequences:
            if len(fgs) > 0:
                device = fgs[0].device
                break

        if device is None:
            device = torch.device("cpu")

        lengths = torch.tensor([len(fgs) for fgs in fg_sequences], dtype=torch.long, device=device)
        max_len = lengths.max().item() if lengths.numel() > 0 and lengths.max().item() > 0 else 1

        batch_size = len(fg_sequences)
        padded = torch.zeros(batch_size, max_len, self.embed_dim, device=device)

        for i, fgs in enumerate(fg_sequences):
            if len(fgs) > 0:
                fg_tensor = torch.stack(fgs)
                seq_len = min(len(fgs), max_len)
                padded[i, :seq_len] = fg_tensor[:seq_len]

        return padded, lengths

    def _apply_fg_boost(self, fg_features: List[List[torch.Tensor]], 
                       fg_info: List[List[dict]]) -> List[List[torch.Tensor]]:
        """对目标官能团应用权重增强"""
        if self.target_fg_boost == 1.0 or not fg_info:
            return fg_features

        boosted_features = []
        for mol_fgs, mol_info in zip(fg_features, fg_info):
            boosted_mol_fgs = []
            for fg_feat, info in zip(mol_fgs, mol_info):
                if info['name'] in self.target_fg_names:
                    boosted_mol_fgs.append(fg_feat * self.target_fg_boost)
                else:
                    boosted_mol_fgs.append(fg_feat)
            boosted_features.append(boosted_mol_fgs)
        
        return boosted_features

    def forward(self,
                v_mol_nodes: torch.Tensor,
                mol_batch_indices: List[torch.Tensor],
                v_fg_features: List[List[torch.Tensor]],
                v_e3_tokens: List[torch.Tensor] = None,
                v_tgt_tokens: List[torch.Tensor] = None,
                v_e3_pooled: torch.Tensor = None,
                v_tgt_pooled: torch.Tensor = None,
                return_attention: bool = False,
                fg_info: List[List[dict]] = None):
        """
        池化特征注意力融合前向传播

        参数:
            v_mol_nodes: 原子级别特征（不使用）
            mol_batch_indices: 原子索引（不使用）
            v_fg_features: 每个分子的官能团特征列表
            v_e3_tokens: E3 token特征（不使用）
            v_tgt_tokens: 靶点token特征（不使用）
            v_e3_pooled: E3池化特征 [B, embed_dim]
            v_tgt_pooled: 靶点池化特征 [B, embed_dim]
            return_attention: 是否返回注意力权重
            fg_info: 官能团信息

        返回:
            h_final: 最终融合特征 [B, embed_dim]
            h_stage1: 阶段1特征 [B, embed_dim]
            (可选) fg_attention_stage1, fg_attention_stage2, fg_info
        """
        batch_size = len(v_fg_features) if v_fg_features else 0
        device = next(self.stage1_attention.parameters()).device

        # 确保有E3和靶点特征
        if v_e3_pooled is None or v_tgt_pooled is None:
            raise ValueError("PooledAttentionFusion需要v_e3_pooled和v_tgt_pooled参数")

        # 应用官能团增强
        if fg_info:
            v_fg_features = self._apply_fg_boost(v_fg_features, fg_info)

        # 1. Padding官能团特征
        fg_padded, fg_lengths = self._pad_fg_sequences(v_fg_features)  # [B, max_N_fg, D]
        
        if fg_padded.size(0) == 0:
            # 如果没有官能团，返回零向量
            zero_out = torch.zeros(batch_size, self.embed_dim, device=device)
            if return_attention:
                return zero_out, zero_out, None, None, fg_info
            else:
                return zero_out, zero_out

        # 创建padding mask
        fg_mask = torch.arange(fg_padded.size(1), device=device)[None, :] >= fg_lengths[:, None]  # [B, max_N_fg]

        # 2. 扩展E3池化特征为"伪token序列"
        # [B, embed_dim] -> [B, 4*embed_dim] -> reshape -> [B, 4, embed_dim]
        e3_expanded = self.e3_expander(v_e3_pooled)  # [B, 4*embed_dim]
        e3_keys = e3_expanded.view(batch_size, 4, self.embed_dim)  # [B, 4, embed_dim]

        # 3. 阶段1: 官能团 attend to E3特征
        # Query: 官能团特征 [B, max_N_fg, D]
        # Key/Value: E3扩展特征 [B, 4, D]
        attn_out1, attn_weights1 = self.stage1_attention(
            query=fg_padded,
            key=e3_keys,
            value=e3_keys,
            key_padding_mask=None,  # E3特征没有padding
            need_weights=return_attention,
            average_attn_weights=True
        )  # attn_out1: [B, max_N_fg, D], attn_weights1: [B, max_N_fg, 4]

        # 残差连接和归一化
        h_stage1_fg = self.stage1_norm(fg_padded + attn_out1)
        h_stage1_fg = h_stage1_fg + self.stage1_ffn(h_stage1_fg)

        # 池化阶段1特征
        h_stage1_pooled = []
        for i in range(batch_size):
            seq_len = fg_lengths[i].item()
            if seq_len > 0:
                h_stage1_pooled.append(h_stage1_fg[i, :seq_len].mean(dim=0))
            else:
                h_stage1_pooled.append(torch.zeros(self.embed_dim, device=device))
        h_stage1_out = torch.stack(h_stage1_pooled, dim=0)  # [B, D]

        # 4. 扩展靶点池化特征为"伪token序列"
        tgt_expanded = self.tgt_expander(v_tgt_pooled)  # [B, 4*embed_dim]
        tgt_keys = tgt_expanded.view(batch_size, 4, self.embed_dim)  # [B, 4, embed_dim]

        # 5. 阶段2: 官能团-E3复合物 attend to 靶点特征
        attn_out2, attn_weights2 = self.stage2_attention(
            query=h_stage1_fg,
            key=tgt_keys,
            value=tgt_keys,
            key_padding_mask=None,
            need_weights=return_attention,
            average_attn_weights=True
        )  # attn_out2: [B, max_N_fg, D], attn_weights2: [B, max_N_fg, 4]

        # 残差连接和归一化
        h_stage2_fg = self.stage2_norm(h_stage1_fg + attn_out2)
        h_stage2_fg = h_stage2_fg + self.stage2_ffn(h_stage2_fg)

        # 池化最终特征
        h_final_pooled = []
        for i in range(batch_size):
            seq_len = fg_lengths[i].item()
            if seq_len > 0:
                h_final_pooled.append(h_stage2_fg[i, :seq_len].mean(dim=0))
            else:
                h_final_pooled.append(torch.zeros(self.embed_dim, device=device))
        h_final = torch.stack(h_final_pooled, dim=0)  # [B, D]

        if return_attention:
            # 返回官能团注意力权重（简化版，只返回平均权重）
            # 注意：这里的注意力是 [B, max_N_fg, 4]，表示每个官能团对4个E3/靶点"伪token"的注意力
            fg_attention_stage1 = attn_weights1[0] if batch_size > 0 else None  # [max_N_fg, 4]
            fg_attention_stage2 = attn_weights2[0] if batch_size > 0 else None  # [max_N_fg, 4]
            return h_final, h_stage1_out, fg_attention_stage1, fg_attention_stage2, fg_info
        else:
            return h_final, h_stage1_out
