from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion using Transformer Decoder for token-level feature fusion.

    Implements sequential interaction: molecular nodes -> E3 protein tokens -> target protein tokens
    Enables attention to specific functional groups and amino acid residues at token level.
    """

    def __init__(self, embed_dim: int, num_heads: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Stage 1: Molecular nodes cross-attend to E3 protein tokens
        decoder_layer_1 = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for stability
        )
        self.decoder_stage1 = nn.TransformerDecoder(decoder_layer_1, num_layers=num_layers)

        # Stage 2: Molecular-E3 complex cross-attends to target protein tokens
        decoder_layer_2 = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder_stage2 = nn.TransformerDecoder(decoder_layer_2, num_layers=num_layers)

    def _compute_cross_attention_weights(self,
                                         query: torch.Tensor,
                                         key: torch.Tensor,
                                         value: torch.Tensor,
                                         key_padding_mask: Optional[torch.Tensor] = None,
                                         layer_idx: int = 0,
                                         stage: int = 1) -> torch.Tensor:
        """
        Manually compute cross-attention weights.

        Args:
            query: [B, N_q, D] query tensor (molecular nodes)
            key: [B, N_k, D] key tensor (E3 or target tokens)
            value: [B, N_k, D] value tensor (E3 or target tokens)
            key_padding_mask: [B, N_k] padding mask
            layer_idx: layer index to use
            stage: stage (1 or 2)

        Returns:
            attention_weights: [B, num_heads, N_q, N_k] attention weights
        """
        B, N_q, D = query.shape
        _, N_k, _ = key.shape

        # Get attention module for corresponding layer
        if stage == 1:
            decoder = self.decoder_stage1
        else:
            decoder = self.decoder_stage2

        # Get cross-attention module from first layer
        decoder_layer = decoder.layers[layer_idx]
        cross_attn = decoder_layer.multihead_attn

        # PyTorch MultiheadAttention uses in_proj_weight and in_proj_bias
        # in_proj_weight: [3*embed_dim, embed_dim] (Q, K, V concatenated)
        if hasattr(cross_attn, 'in_proj_weight') and cross_attn.in_proj_weight is not None:
            in_proj_weight = cross_attn.in_proj_weight 
            in_proj_bias = cross_attn.in_proj_bias if cross_attn.in_proj_bias is not None else None

            # Split weights and biases
            q_proj_weight = in_proj_weight[:self.embed_dim, :]
            k_proj_weight = in_proj_weight[self.embed_dim:2*self.embed_dim, :]
    
            if in_proj_bias is not None:
                q_bias = in_proj_bias[:self.embed_dim]
                k_bias = in_proj_bias[self.embed_dim:2*self.embed_dim]
            else:
                q_bias = None
                k_bias = None
        else:
            # If using separate weights
            q_proj_weight = cross_attn.q_proj_weight
            k_proj_weight = cross_attn.k_proj_weight
            q_bias = cross_attn.in_proj_bias[:self.embed_dim] if cross_attn.in_proj_bias is not None else None
            k_bias = cross_attn.in_proj_bias[self.embed_dim:2*self.embed_dim] if cross_attn.in_proj_bias is not None else None

        q = F.linear(query, q_proj_weight, q_bias) 
        k = F.linear(key, k_proj_weight, k_bias)   

        head_dim = self.embed_dim // self.num_heads
        q = q.view(B, N_q, self.num_heads, head_dim).transpose(1, 2)  
        k = k.view(B, N_k, self.num_heads, head_dim).transpose(1, 2) 


        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)  # [B, num_heads, N_q, N_k]

        # Apply padding mask
        if key_padding_mask is not None:
            # key_padding_mask: [B, N_k], True indicates positions to mask
            # Expand to [B, num_heads, N_q, N_k]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N_k]
            scores = scores.masked_fill(mask, float('-inf'))

        # Softmax
        attention_weights = F.softmax(scores, dim=-1)  # [B, num_heads, N_q, N_k]

        return attention_weights

    def _pad_sequences(self, sequences: List[torch.Tensor], batch_first: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad variable-length sequences to same length.

        Args:
            sequences: List of sequences, each [L_i, D]
        Returns:
            padded: Padded tensor [B, max_L, D]
            lengths: Actual lengths of each sequence [B]
        """
        if len(sequences) == 0:
            device = torch.device("cpu")
            return torch.empty(0, 0, self.embed_dim, device=device), torch.empty(0, dtype=torch.long, device=device)

        # Get device
        device = sequences[0].device

        # Create lengths tensor on correct device
        lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long, device=device)
        max_len = lengths.max().item()

        # Create padded tensor
        batch_size = len(sequences)
        padded = torch.zeros(batch_size, max_len, self.embed_dim, device=device)

        for i, seq in enumerate(sequences):
            seq_len = seq.size(0)
            padded[i, :seq_len] = seq

        return padded, lengths

    def forward(self,
                v_mol_nodes: torch.Tensor,
                mol_batch_indices: List[torch.Tensor],
                v_e3_tokens: List[torch.Tensor],
                v_tgt_tokens: List[torch.Tensor],
                return_attention: bool = False):
        """
        Token-level cross-attention forward pass.

        Args:
            v_mol_nodes: Batched molecular node features [N_total, D]
            mol_batch_indices: Node indices for each molecule in batch, length B
            v_e3_tokens: E3 protein token features, each [L_i, D]
            v_tgt_tokens: Target protein token features, each [L_i, D]
            return_attention: Whether to return attention weights

        Returns:
            If return_attention=False:
                h_final: Final fused features [B, D]
                h_stage1: Stage 1 output (mol-E3) [B, D]
            If return_attention=True:
                h_final, h_stage1, attention_stage1, attention_stage2
        """
        batch_size = len(mol_batch_indices)
        device = v_mol_nodes.device

        # Extract node features for each molecule
        mol_sequences = []
        for i in range(batch_size):
            node_indices = mol_batch_indices[i]
            if node_indices.numel() > 0:
                mol_nodes = v_mol_nodes[node_indices]  # [N_i, D]
            else:
                # Create dummy node if molecule has no valid nodes
                mol_nodes = torch.zeros(1, self.embed_dim, device=device)
            mol_sequences.append(mol_nodes)

        # Pad molecular node sequences
        mol_padded, mol_lengths = self._pad_sequences(mol_sequences)  # [B, max_N, D]

        # Pad E3 protein sequences
        e3_padded, e3_lengths = self._pad_sequences(v_e3_tokens)  # [B, max_L_e3, D]

        # Pad target protein sequences
        tgt_padded, tgt_lengths = self._pad_sequences(v_tgt_tokens)  # [B, max_L_tgt, D]

        # Create attention masks to avoid padding positions in computation
        mol_mask = torch.arange(mol_padded.size(1), device=device)[None, :] >= mol_lengths[:, None]  # [B, max_N]
        e3_mask = torch.arange(e3_padded.size(1), device=device)[None, :] >= e3_lengths[:, None]  # [B, max_L_e3]
        tgt_mask = torch.arange(tgt_padded.size(1), device=device)[None, :] >= tgt_lengths[:, None]  # [B, max_L_tgt]

        # Extract attention weights if requested
        attention_stage1 = None
        attention_stage2 = None

        if return_attention:
            # Compute stage 1 attention weights (using first layer)
            attention_stage1 = self._compute_cross_attention_weights(
                query=mol_padded,
                key=e3_padded,
                value=e3_padded,
                key_padding_mask=e3_mask,
                layer_idx=0,
                stage=1
            )  # [B, num_heads, max_N, max_L_e3]

        # Stage 1: Molecular nodes cross-attend to E3 protein tokens
        # Each molecule's nodes are enhanced by attending to E3 protein tokens
        # This allows learning which functional groups relate to which E3 amino acids
        h_stage1 = self.decoder_stage1(
            tgt=mol_padded,
            memory=e3_padded,
            tgt_key_padding_mask=mol_mask,
            memory_key_padding_mask=e3_mask
        )  # [B, max_N, D]

        if return_attention:
            # Compute stage 2 attention weights (using first layer)
            attention_stage2 = self._compute_cross_attention_weights(
                query=h_stage1,
                key=tgt_padded,
                value=tgt_padded,
                key_padding_mask=tgt_mask,
                layer_idx=0,
                stage=2
            )  # [B, num_heads, max_N, max_L_tgt]

        # Stage 2: Molecular-E3 complex cross-attends to target protein tokens
        # Enhanced molecular nodes attend to target protein tokens for final enhancement
        # This allows learning which mol-E3 complex parts relate to target amino acids
        h_stage2 = self.decoder_stage2(
            tgt=h_stage1,
            memory=tgt_padded,
            tgt_key_padding_mask=mol_mask,
            memory_key_padding_mask=tgt_mask
        )  # [B, max_N, D]

        # Pool node features for each molecule (only valid nodes)
        h_stage1_pooled = []
        h_final_pooled = []

        for i in range(batch_size):
            seq_len = mol_lengths[i].item()
            if seq_len > 0:
                # Average pool only valid nodes
                h_stage1_pooled.append(h_stage1[i, :seq_len].mean(dim=0))  # [D]
                h_final_pooled.append(h_stage2[i, :seq_len].mean(dim=0))  # [D]
            else:
                # Use zero vector if no valid nodes
                h_stage1_pooled.append(torch.zeros(self.embed_dim, device=device))
                h_final_pooled.append(torch.zeros(self.embed_dim, device=device))

        h_stage1_out = torch.stack(h_stage1_pooled, dim=0)  # [B, D]
        h_final = torch.stack(h_final_pooled, dim=0)         # [B, D]

        if return_attention:
            # Process attention weights: keep only valid parts
            if attention_stage1 is not None:
                # Take first sample (batch_idx=0) valid parts only
                if batch_size > 0:
                    mol_len = mol_lengths[0].item()
                    e3_len = e3_lengths[0].item()
                    attn1 = attention_stage1[0].max(dim=0)[0][:mol_len, :e3_len]
                    attention_stage1 = attn1
                else:
                    attention_stage1 = None

            if attention_stage2 is not None:
                # Take first sample (batch_idx=0) valid parts only
                if batch_size > 0:
                    mol_len = mol_lengths[0].item()
                    tgt_len = tgt_lengths[0].item()
                    attn2 = attention_stage2[0].max(dim=0)[0][:mol_len, :tgt_len]
                    attention_stage2 = attn2
                else:
                    attention_stage2 = None

            return h_final, h_stage1_out, attention_stage1, attention_stage2
        else:
            return h_final, h_stage1_out


class ConcatenationFusion(nn.Module):
    """
    Direct concatenation fusion: Concatenates mol, E3, target features for ablation study.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Feature projection after concatenation
        # Input: mol + E3 + target features = 3 * embed_dim
        # Output: embed_dim
        self.concat_projection = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

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
        Direct concatenation forward pass (ablation study).

        Args:
            v_mol_nodes: Atom-level features (not used)
            mol_batch_indices: Atom indices (not used)
            v_fg_features: Functional group features per molecule
            v_e3_tokens: E3 protein token features (optional if v_e3_pooled provided)
            v_tgt_tokens: Target protein token features (optional if v_tgt_pooled provided)
            v_e3_pooled: Pooled E3 features [B, embed_dim] (preferred if available)
            v_tgt_pooled: Pooled target features [B, embed_dim] (preferred if available)
            return_attention: Whether to return attention (always None here)
            fg_info: Functional group info (not used)

        Returns:
            h_final: Final fused features [B, D]
            h_stage1: Stage 1 features [B, D] (same as h_final)
        """
        batch_size = len(v_fg_features) if v_fg_features else 0
        device = next(self.concat_projection.parameters()).device

        # 1. Pool molecular functional group features
        mol_features = []
        for fg_list in v_fg_features:
            if fg_list and len(fg_list) > 0:
                fg_tensor = torch.stack(fg_list)  # [num_fg, embed_dim]
                mol_feat = fg_tensor.mean(dim=0)  # [embed_dim]
            else:
                mol_feat = torch.zeros(self.embed_dim, device=device)
            mol_features.append(mol_feat)
        mol_pooled = torch.stack(mol_features, dim=0)  # [B, embed_dim]

        # 2. Get E3 protein features (prefer pooled features)
        if v_e3_pooled is not None:
            e3_pooled = v_e3_pooled  # [B, embed_dim]
        elif v_e3_tokens is not None:
            e3_features = []
            for e3_tokens in v_e3_tokens:
                if e3_tokens.numel() > 0:
                    e3_feat = e3_tokens.mean(dim=0)  # [embed_dim]
                else:
                    e3_feat = torch.zeros(self.embed_dim, device=device)
                e3_features.append(e3_feat)
            e3_pooled = torch.stack(e3_features, dim=0)  # [B, embed_dim]
        else:
            e3_pooled = torch.zeros(batch_size, self.embed_dim, device=device)

        # 3. Get target protein features (prefer pooled features)
        if v_tgt_pooled is not None:
            tgt_pooled = v_tgt_pooled  # [B, embed_dim]
        elif v_tgt_tokens is not None:
            tgt_features = []
            for tgt_tokens in v_tgt_tokens:
                if tgt_tokens.numel() > 0:
                    tgt_feat = tgt_tokens.mean(dim=0)  # [embed_dim]
                else:
                    tgt_feat = torch.zeros(self.embed_dim, device=device)
                tgt_features.append(tgt_feat)
            tgt_pooled = torch.stack(tgt_features, dim=0)  # [B, embed_dim]
        else:
            tgt_pooled = torch.zeros(batch_size, self.embed_dim, device=device)

        # 4. Directly concatenate three feature sets
        concat_features = torch.cat([mol_pooled, e3_pooled, tgt_pooled], dim=-1)  # [B, 3*embed_dim]

        # 5. Project to final dimension
        h_final = self.concat_projection(concat_features)  # [B, embed_dim]
        h_stage1 = h_final  # Stage 1 features same as final in ablation

        if return_attention:
            # Concatenation method has no attention weights, return None
            return h_final, h_stage1, None, None, fg_info
        else:
            return h_final, h_stage1


class MultiScaleAttentionFusion(nn.Module):
    """
    Functional group attention fusion: Uses only functional group-level attention, skips atomic level.
    """

    def __init__(self, embed_dim: int, num_heads: int, num_layers: int = 2, dropout: float = 0.1,
                 enable_fg_boost: bool = True, fg_boost_factor: float = 2.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.target_fg_names = {'cyclic_imide_1', 'cyclic_imide_2', 'cyclic_imide_3'}
        self.target_fg_boost = fg_boost_factor if enable_fg_boost else 1.0

        # Use only functional group-level attention
        self.fg_attention = CrossAttentionFusion(embed_dim, num_heads, num_layers, dropout)

    def _pad_fg_sequences(self, fg_sequences: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad variable-length functional group sequences"""
        if len(fg_sequences) == 0 or all(len(fgs) == 0 for fgs in fg_sequences):
            device = torch.device("cpu")
            if len(fg_sequences) > 0:
                # Try to get device
                for fgs in fg_sequences:
                    if len(fgs) > 0:
                        device = fgs[0].device
                        break
            return torch.empty(0, 0, self.embed_dim, device=device), torch.empty(0, dtype=torch.long, device=device)

        # Get device
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
                fg_tensor = torch.stack(fgs)  # [num_fgs, embed_dim]
                seq_len = min(len(fgs), max_len)
                padded[i, :seq_len] = fg_tensor[:seq_len]

        return padded, lengths

    def forward(self,
                v_mol_nodes: torch.Tensor,
                mol_batch_indices: List[torch.Tensor],
                v_fg_features: List[List[torch.Tensor]],
                v_e3_tokens: List[torch.Tensor],
                v_tgt_tokens: List[torch.Tensor],
                return_attention: bool = False,
                fg_info: List[List[dict]] = None):
        """
        Functional group attention forward pass (functional group-level attention only).

        Args:
            v_mol_nodes: Atomic-level features [N_total, D] (not used)
            mol_batch_indices: Atomic indices per molecule (not used)
            v_fg_features: Functional group features per molecule
            v_e3_tokens: E3 protein token features
            v_tgt_tokens: Target protein token features
            return_attention: Whether to return attention weights
            fg_info: Functional group info (for visualization)

        Returns:
            h_final: Final fused features [B, D]
            h_stage1: Stage 1 features [B, D]
            (If return_attention=True, also returns FG attention weights and info)
        """
        batch_size = len(v_fg_features) if v_fg_features else 0

        # Get device
        device = next(self.fg_attention.parameters()).device
        if v_fg_features and len(v_fg_features) > 0 and len(v_fg_features[0]) > 0:
            device = v_fg_features[0][0].device if len(v_fg_features[0]) > 0 else device

        # 1. Boost target functional group feature weights
        fg_padded, fg_lengths = self._pad_fg_sequences(v_fg_features)
        if fg_info is not None and fg_lengths.numel() > 0:
            max_batch = min(batch_size, len(fg_info))
            for i in range(max_batch):
                current_fg_info = fg_info[i] if i < len(fg_info) else []
                if fg_lengths.numel() <= i:
                    continue
                fg_len = fg_lengths[i].item()
                if fg_len <= 0 or len(current_fg_info) == 0:
                    continue
                limit = min(fg_len, len(current_fg_info))
                for j in range(limit):
                    fg_name = current_fg_info[j].get('name', '')
                    if fg_name in self.target_fg_names:
                        fg_padded[i, j] = fg_padded[i, j] * self.target_fg_boost

        # 2. Functional group-level attention
        attn_fg_stage1 = None
        attn_fg_stage2 = None

        if fg_padded.size(1) > 0 and fg_lengths.max().item() > 0:  # If functional groups exist
            # Create batch indices for FGs (each molecule's FGs as independent sequence)
            fg_sequences = []
            device_fg = fg_padded.device

            for i in range(batch_size):
                fg_len = fg_lengths[i].item()
                if fg_len > 0:
                    fg_seq = fg_padded[i, :fg_len]  # [fg_len, embed_dim]
                    fg_sequences.append(fg_seq)
                else:
                    # Create dummy zero vector if no FGs
                    dummy_fg = torch.zeros(1, self.embed_dim, device=device_fg)
                    fg_sequences.append(dummy_fg)

            # Flatten all FGs for batching
            if len(fg_sequences) > 0:
                fg_flat_list = []
                fg_batch_idx_list = []
                node_start = 0
                for fg_seq in fg_sequences:
                    fg_flat_list.append(fg_seq)
                    fg_batch_idx_list.append(torch.arange(node_start, node_start + len(fg_seq), device=device_fg))
                    node_start += len(fg_seq)

                fg_flat = torch.cat(fg_flat_list, dim=0)  # [total_fg, embed_dim]

                fg_output = self.fg_attention(
                    v_mol_nodes=fg_flat,
                    mol_batch_indices=fg_batch_idx_list,
                    v_e3_tokens=v_e3_tokens,
                    v_tgt_tokens=v_tgt_tokens,
                    return_attention=return_attention
                )

                # Process fg_output
                if isinstance(fg_output, tuple) and len(fg_output) >= 2:
                    h_fg_final, h_fg_stage1 = fg_output[:2]
                    # Check if attention weights exist
                    if len(fg_output) >= 3 and return_attention:
                        attn_fg_stage1, attn_fg_stage2 = fg_output[2:4] if len(fg_output) >= 4 else (fg_output[2], None)

                        # Apply boosting logic
                        if attn_fg_stage1 is not None and fg_info:
                            target_indices = []
                            first_fg_info = fg_info[0] if len(fg_info) > 0 else []
                            for idx, info in enumerate(first_fg_info):
                                fg_name = info.get('name', '')
                                if fg_name in self.target_fg_names:
                                    target_indices.append(idx)

                            for idx in target_indices:
                                if attn_fg_stage1.size(0) > idx:
                                    # Boost without renormalization to maintain absolute differences
                                    attn_fg_stage1[idx, :] = attn_fg_stage1[idx, :] * self.target_fg_boost
                else:
                    h_fg_final = fg_output if not isinstance(fg_output, tuple) else fg_output[0]
                    h_fg_stage1 = h_fg_final
            else:
                h_fg_final = torch.zeros(batch_size, self.embed_dim, device=device)
                h_fg_stage1 = torch.zeros(batch_size, self.embed_dim, device=device)
        else:
            # Use zero vectors if no functional groups
            h_fg_final = torch.zeros(batch_size, self.embed_dim, device=device)
            h_fg_stage1 = torch.zeros(batch_size, self.embed_dim, device=device)

        # 3. Use functional group-level features directly (no atomic feature fusion)
        h_final = h_fg_final
        h_stage1 = h_fg_stage1

        if return_attention:
            # Return functional group attention weights and info
            return h_final, h_stage1, attn_fg_stage1, attn_fg_stage2, fg_info
        else:
            return h_final, h_stage1
