from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

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
                 # GNN molecule encoder configuration
                 gnn_type: str = "gcn",  # "gcn" or "gat"
                 gnn_layers: int = 3,  # Number of GNN layers
                 gnn_hidden_dim: int = 128,  # GNN hidden layer dimension
                 node_feat_dim: int = 44,  # Node feature dimension
                 edge_feat_dim: int = 10,  # Edge feature dimension
                 # Functional group attention configuration
                 use_multiscale_attention: bool = True,  # Whether to use multi-scale attention
                 custom_functional_groups: dict = None,  # Custom functional group SMARTS patterns
                 use_fg_only_attention: bool = True,  # Whether to use only functional group-level attention (skip atom-level)
                 enable_fg_boost: bool = True,  # Whether to enable functional group boost (use 2x weight for target functional groups)
                 fusion_method: str = "attention"):  # Fusion method: "attention" or "concat"
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

        # 1. Protein sequence encoder (ESMC）
        self.seq_encoder = SequenceEncoder(
            device=device,
            proj_method=proj_method,
            embed_dim=embed_dim,
            dropout=dropout
        )

        # 2. Molecule encoder
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
        
        # 3. Feature fusion module
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
            raise ValueError(f"Unsupported fusion method: {fusion_method}")
        
        # 4. Prediction head
        hidden_dims = [int(x) for x in mlp_hidden.split(',')]
        
        # Input: final fused features + stage 1 features
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
        layers.append(nn.Linear(prev_dim, 1)) 
        
        self.prediction_head = nn.Sequential(*layers)
        
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

        # 3. Feature fusion module
        if self.fusion_method == "attention":
            fusion_output = self.fusion(
                v_mol_nodes=None, 
                mol_batch_indices=None, 
                v_fg_features=v_fg_features,
                v_e3_tokens=v_e3_tokens,
                v_tgt_tokens=v_tgt_tokens,
                return_attention=return_attention,
                fg_info=fg_info
            )
        else:
            # concat
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
            attention_stage1 = None  
            attention_stage2 = None  
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
            return score, fg_attention_stage1, fg_attention_stage2, fg_info
        else:
            return score
