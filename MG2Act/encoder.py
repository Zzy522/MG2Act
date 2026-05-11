"""
Unified encoder module
Contains molecular encoder and protein sequence encoder
"""

from typing import List, Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

RDKIT_AVAILABLE = False
TORCH_GEOMETRIC_AVAILABLE = False

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    pass

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, GATConv
    from torch_geometric.nn import global_mean_pool, global_max_pool
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    pass

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProteinTensor, LogitsConfig
from esm.tokenization import EsmSequenceTokenizer

from .utils import LayerNorm, ConvPooler


class FunctionalGroupDetector:
    """Functional group detector with custom SMARTS patterns (optimized: pre-compiled SMARTS)"""

    def __init__(self, custom_smarts: dict = None):
        """
        Args:
            custom_smarts: Custom SMARTS patterns dict, format: {'name': 'SMARTS_pattern'}
        """
        # Default common functional groups (extended: includes more N-heterocycles and important groups)
        self.default_smarts = {
            # Basic functional groups
            'amide': '[C;$(C=O)]N',
            'carboxylic_acid': 'C(=O)[O;H]',
            'ester': 'C(=O)[O]',
            'ketone': '[C;$(C=O)]',
            'hydroxyl': '[OX2H]',
            'aromatic_6ring': 'c1ccccc1',
            'aromatic_5ring': 'c1cccc1',
            'nitrile': 'C#N',
            'halide+': '[F,Cl,Br,I,S]',
            'pyridine': 'c1ccncc1',
            'pyrrole': 'c1cc[nH]c1',
            'pyrrolidine': 'C1CCCN1',
            'furan': 'c1ccoc1'
        }

        # Merge custom SMARTS, clear cache
        self.smarts_patterns = self.default_smarts.copy()
        if custom_smarts:
            self.smarts_patterns.update(custom_smarts)

        self.compiled_patterns = {}
        for name, smarts in self.smarts_patterns.items():
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern:
                    self.compiled_patterns[name] = pattern
            except Exception as e:
                continue

    def extract_functional_groups(self, mol):
        """
        Extract all functional groups from molecule (using pre-compiled patterns for performance).

        Returns:
            List[Dict]: Each contains {'name': str, 'atoms': tuple, 'center_atom': int, 'num_atoms': int}
        """
        if mol is None:
            return []

        functional_groups = []

        for name, pattern in self.compiled_patterns.items():
            try:
                matches = mol.GetSubstructMatches(pattern)
                for match in matches:
                    center_atom = match[0] if len(match) > 0 else None
                    functional_groups.append({
                        'name': name,
                        'atoms': match,
                        'center_atom': center_atom,
                        'num_atoms': len(match)
                    })
            except Exception as e:
                continue

        return functional_groups


class GNNMolecularEncoder(nn.Module):
    """
    GNN molecular encoder (returns token-level features).

    Converts SMILES to molecular graph, uses GNN to learn representations for each atom (node).
    Returns features for each atom rather than graph-level pooled single feature.
    Enables token-level attention mechanisms to focus on specific functional groups.
    """
    def __init__(self,
                 node_feat_dim: int = 44, 
                 edge_feat_dim: int = 10, 
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 output_dim: int = 64, 
                 gnn_type: str = "gcn",
                 dropout: float = 0.1):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gnn_type = gnn_type.lower()
        self.num_layers = num_layers

        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is not installed. Cannot use the GNN encoder. Please run: pip install rdkit")

        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric is not installed. Cannot use the GNN encoder. Please run: pip install torch-geometric")

        self.node_embed = nn.Linear(node_feat_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        if self.gnn_type == "gcn":
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        elif self.gnn_type == "gat":
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 2):
            if self.gnn_type == "gcn":
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif self.gnn_type == "gat":
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        if num_layers > 1:
            if self.gnn_type == "gcn":
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif self.gnn_type == "gat":
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.node_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def smiles_to_graph(self, smiles: str, device: torch.device):
        """
        Convert SMILES to PyTorch Geometric Data object.

        Returns:
            Data(x, edge_index, edge_attr) or None (if SMILES invalid)
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Atomic features (node features)
            node_features = []
            for atom in mol.GetAtoms():
                features = [
                    atom.GetAtomicNum(),  # Atomic number
                    atom.GetDegree(),  # Degree
                    atom.GetFormalCharge(),  # Formal charge
                    int(atom.GetHybridization()),  # Hybridization type
                    int(atom.GetIsAromatic()),  # Is aromatic
                    atom.GetNumRadicalElectrons(),  # Radical electrons
                    atom.GetNumImplicitHs(),  # Implicit hydrogens
                    int(atom.IsInRing()),  # Is in ring
                ]

                # One-hot encoding for common atom types
                atom_types = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # H, C, N, O, F, P, S, Cl, Br, I
                atom_type_onehot = [int(atom.GetAtomicNum() == t) for t in atom_types]
                features.extend(atom_type_onehot)

                # Pad to fixed dimension
                while len(features) < self.node_feat_dim:
                    features.append(0)
                features = features[:self.node_feat_dim]

                node_features.append(features)

            if len(node_features) == 0:
                return None

            x = torch.tensor(node_features, dtype=torch.float, device=device)

            # Edges (chemical bonds)
            edge_indices = []
            edge_features = []

            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                # Bidirectional edges
                edge_indices.append([i, j])
                edge_indices.append([j, i])

                # Bond features
                bond_type = bond.GetBondType()
                bond_features = [
                    float(bond_type == Chem.BondType.SINGLE),
                    float(bond_type == Chem.BondType.DOUBLE),
                    float(bond_type == Chem.BondType.TRIPLE),
                    float(bond_type == Chem.BondType.AROMATIC),
                    float(bond.GetIsConjugated()),
                    float(bond.IsInRing()),
                ]

                # Pad to fixed dimension
                while len(bond_features) < self.edge_feat_dim:
                    bond_features.append(0.0)
                bond_features = bond_features[:self.edge_feat_dim]

                edge_features.append(bond_features)
                edge_features.append(bond_features)  # Same features for bidirectional edges

            if len(edge_indices) == 0:
                # Create self-loops if no edges
                num_nodes = len(node_features)
                edge_indices = [[i, i] for i in range(num_nodes)]
                edge_features = [[0.0] * self.edge_feat_dim] * num_nodes

            edge_index = torch.tensor(edge_indices, dtype=torch.long, device=device).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float, device=device)

            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        except Exception as e:
            # Silently handle SMILES conversion failures (reduce IO overhead)
            # Uncomment for debugging: print(f"Warning: SMILES conversion failed ({smiles[:50]}): {e}")
            return None

    def forward(self, smiles_list: List[str]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Encode SMILES list, return token-level features.

        Args:
            smiles_list: List of SMILES strings [B]
        Returns:
            node_features: Batched node features [N_total, output_dim]
            batch_indices: Node indices for each molecule in batch, length B
        """
        device = next(self.node_projection.parameters()).device

        # Convert to graph data
        graphs = []
        valid_indices = []

        for idx, smiles in enumerate(smiles_list):
            graph = self.smiles_to_graph(smiles, device)
            if graph is not None:
                graphs.append(graph)
                valid_indices.append(idx)

        if len(graphs) == 0:
            # Return zero vectors if all SMILES invalid
            batch_size = len(smiles_list)
            dummy_node = torch.zeros(1, self.output_dim, device=device)
            empty_indices = [torch.tensor([], dtype=torch.long, device=device) for _ in range(batch_size)]
            return dummy_node, empty_indices

        batch = Batch.from_data_list(graphs)
        x = batch.x  # [N_total, node_feat_dim]
        edge_index = batch.edge_index  # [2, E_total]
        batch_idx = batch.batch  # [N_total]

        # Node feature embedding
        x = self.node_embed(x)  # [N_total, hidden_dim]

        # GNN forward pass
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.1, training=self.training)

        # Project to output dimension (per node)
        node_features = self.node_projection(x)  # [N_total, output_dim]

        # Build node index list for each molecule
        batch_indices = []
        node_start = 0
        for i, graph in enumerate(graphs):
            num_nodes = graph.x.size(0)
            node_end = node_start + num_nodes
            batch_indices.append(torch.arange(node_start, node_end, device=device))
            node_start = node_end

        # Fill empty indices if some SMILES were invalid
        if len(valid_indices) < len(smiles_list):
            full_batch_indices = []
            valid_idx_map = {idx: batch_indices[i] for i, idx in enumerate(valid_indices)}
            for i in range(len(smiles_list)):
                if i in valid_idx_map:
                    full_batch_indices.append(valid_idx_map[i])
                else:
                    full_batch_indices.append(torch.tensor([], dtype=torch.long, device=device))
            batch_indices = full_batch_indices

        return node_features, batch_indices


class EnhancedGNNMolecularEncoder(nn.Module):
    """
    Enhanced GNN molecular encoder: Outputs both atomic-level and functional group-level features.
    """

    def __init__(self,
                 node_feat_dim: int = 44,
                 edge_feat_dim: int = 10,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 output_dim: int = 64,
                 gnn_type: str = "gat",
                 dropout: float = 0.1,
                 custom_functional_groups: dict = None):
        super().__init__()

        self.atom_encoder = GNNMolecularEncoder(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            gnn_type=gnn_type,
            dropout=dropout
        )

        self.fg_detector = FunctionalGroupDetector(custom_functional_groups)

        self.fg_encoders = nn.ModuleDict()
        for fg_name in self.fg_detector.smarts_patterns.keys():
            self.fg_encoders[fg_name] = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        self.fg_encoder_generic = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, smiles_list: List[str]) -> Tuple[torch.Tensor, List[torch.Tensor], List[List[torch.Tensor]], List[List[dict]]]:
        """
        Returns:
            atom_features: [N_total, output_dim] atomic-level features
            mol_batch_indices: List[torch.Tensor] atomic indices per molecule
            functional_group_features: List[List[torch.Tensor]] functional group features per molecule
            functional_group_info: List[List[Dict]] functional group info per molecule
        """
        device = next(self.atom_encoder.parameters()).device

        # 1. Get atomic-level features
        atom_features, mol_batch_indices = self.atom_encoder(smiles_list)

        # 2. Extract functional group features
        functional_group_features = []
        functional_group_info = []

        for idx, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                functional_group_features.append([])
                functional_group_info.append([])
                continue

            # Detect functional groups
            fg_list = self.fg_detector.extract_functional_groups(mol)

            # Get atom indices for this molecule
            atom_indices = mol_batch_indices[idx]
            if atom_indices.numel() == 0:
                functional_group_features.append([])
                functional_group_info.append([])
                continue

            # Extract features for each functional group
            fg_features = []
            fg_info = []

            # Map molecule atom indices to graph node indices
            for fg in fg_list:
                # Get graph indices for atoms in this functional group
                fg_atom_indices_in_graph = []
                for mol_atom_idx in fg['atoms']:
                    if mol_atom_idx < len(atom_indices):
                        graph_node_idx = atom_indices[mol_atom_idx].item()
                        fg_atom_indices_in_graph.append(graph_node_idx)

                if len(fg_atom_indices_in_graph) == 0:
                    continue

                # Pool functional group atom features (average pooling)
                fg_atom_features = atom_features[fg_atom_indices_in_graph]  # [num_atoms_in_fg, output_dim]
                fg_pooled = fg_atom_features.mean(dim=0)  # [output_dim]

                # Encode through corresponding encoder
                fg_name = fg['name']
                if fg_name in self.fg_encoders:
                    fg_encoded = self.fg_encoders[fg_name](fg_pooled)
                else:
                    fg_encoded = self.fg_encoder_generic(fg_pooled)

                fg_features.append(fg_encoded)
                fg_info.append(fg)

            functional_group_features.append(fg_features)
            functional_group_info.append(fg_info)

        return atom_features, mol_batch_indices, functional_group_features, functional_group_info


class SequenceEncoder(nn.Module):
    """
    Protein sequence encoder (based on ESMC), uses ConvPooler for dimensionality reduction.

    Features:
    - Supports both pooled and token-level features
    - Smart caching: Automatically caches encoded sequences to avoid recomputation
    - Caching only during inference to ensure training consistency
    """

    def __init__(self, device: torch.device = torch.device("cpu"),
                 proj_method: str = "conv",  # "conv", "mlp", "linear"
                 embed_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.device = device
        self.proj_method = proj_method
        self.embed_dim = embed_dim

        import os
        os.environ.setdefault("INFRA_PROVIDER", "True")
    
        self.esmc = ESMC.from_pretrained("esmc_300m", device=device)
        self.tokenizer = EsmSequenceTokenizer()

        # Freeze strategy: Freeze all parameters by default
        for p in self.esmc.parameters():
            p.requires_grad = False

        self.any_unfrozen = False
        self.esmc.eval()

        # Cache encoded results: Support both pooled and token-level features
        self.cache = {}  # Format: {seq: {"pooled": tensor, "tokens": tensor}}

        with torch.no_grad():
            tmp_tokens = self.tokenizer.encode("M")
            tmp_tensor = ESMProteinTensor(sequence=torch.tensor(tmp_tokens).to(device))
            tmp_logits = self.esmc.logits(tmp_tensor, LogitsConfig(sequence=True, return_embeddings=True))
            self.d_model = tmp_logits.embeddings.shape[-1]

        self.protein_projector = self._build_projector()
        self.to(self.device)

    def _build_projector(self):
        if self.proj_method == "conv":
            return ConvPooler(self.d_model, self.embed_dim, kernel_size=3, stride=1)
        elif self.proj_method == "mlp":
            return nn.Sequential(
                nn.Linear(self.d_model, self.embed_dim * 4),
                nn.LayerNorm(self.embed_dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.embed_dim * 4, self.embed_dim * 2),
                nn.LayerNorm(self.embed_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                nn.LayerNorm(self.embed_dim)
            )
        else:
            # 简单线性（保持兼容）
            return nn.Linear(self.d_model, self.embed_dim)

    def _encode_sequence(self, seq: str, use_grad: bool = False):
        """Internal method: Encode single sequence, return raw ESMC embeddings"""
        tokens = self.tokenizer.encode(seq)
        prot = ESMProteinTensor(sequence=torch.tensor(tokens).to(self.device))

        if use_grad:
            self.esmc.train()
            logits_out = self.esmc.logits(prot, LogitsConfig(sequence=True, return_embeddings=True))
            emb = logits_out.embeddings
        else:
            self.esmc.eval()
            with torch.no_grad():
                logits_out = self.esmc.logits(prot, LogitsConfig(sequence=True, return_embeddings=True))
                emb = logits_out.embeddings

        return emb.to(self.device)

    def _project_tokens(self, emb: torch.Tensor):
        """Internal method: Project ESMC embeddings at token level"""
        if emb.dim() == 2:  # [L, D]
            if self.proj_method == "conv":
                emb_in = emb.unsqueeze(0)  # [1, L, D]
                projected = self.protein_projector(emb_in)  # [1, L', embed_dim]
                return projected.squeeze(0)  # [L', embed_dim]
            else:
                # MLP/Linear
                return self.protein_projector(emb)  # [L, embed_dim]
        elif emb.dim() == 3:  # [1, L, D]
            if self.proj_method == "conv":
                projected = self.protein_projector(emb)  # [1, L', embed_dim]
                return projected.squeeze(0)  # [L', embed_dim]
            else:
                emb_flat = emb.view(-1, emb.size(-1))  # [L, D]
                return self.protein_projector(emb_flat)  # [L, embed_dim]
        else:
            return self.protein_projector(emb)

    def _pool_embedding(self, emb: torch.Tensor):
        """Internal method: Pool ESMC embeddings"""
        if self.proj_method == "conv":
            # ConvPooler  [B, L, D]
            if emb.dim() == 2:
                emb_in = emb.unsqueeze(0)  # [1, L, D]
            elif emb.dim() == 3 and emb.size(0) == 1:
                emb_in = emb  # [1, L, D]
            else:
                emb_in = emb
            projected = self.protein_projector(emb_in)  # [1, L', embed_dim]
            return projected.mean(dim=1).squeeze(0)  # [embed_dim]
        else:
            # MLP/Linear
            if emb.dim() == 2:
                emb_pooled = emb.mean(dim=0)
            elif emb.dim() == 3:
                emb_pooled = emb.mean(dim=1).squeeze(0)
            else:
                emb_pooled = emb
            return self.protein_projector(emb_pooled)

    def forward(self, seq_list: List[str], return_token_level: bool = False):
        """
        Encode protein sequences.

        Args:
            seq_list: List of protein sequences
            return_token_level: If True, return token-level features [B, L, embed_dim]; else pooled features [B, embed_dim]
        Returns:
            If return_token_level=True: List of token-level features, each [L_i, embed_dim]
            If return_token_level=False: Encoded vectors [B, embed_dim]
        """
        if return_token_level:
            token_outputs = []

            for seq in seq_list:
                use_grad = self.any_unfrozen and self.training

                # Check cache (inference only)
                if not use_grad and seq in self.cache and "tokens" in self.cache[seq]:
                    cached_tokens = self.cache[seq]["tokens"]
                    if cached_tokens.device != self.device:
                        cached_tokens = cached_tokens.to(self.device)
                    token_outputs.append(cached_tokens)
                    continue

                # Encode sequence
                emb = self._encode_sequence(seq, use_grad)
                token_features = self._project_tokens(emb)

                # Cache token-level features (inference only)
                if not use_grad:
                    if seq not in self.cache:
                        self.cache[seq] = {}
                    self.cache[seq]["tokens"] = token_features.detach().cpu()

                token_outputs.append(token_features)

            return token_outputs
        else:
            outputs: List[torch.Tensor] = []

            for seq in seq_list:
                use_grad = self.any_unfrozen and self.training

                # Check cache (inference only)
                if not use_grad and seq in self.cache and "pooled" in self.cache[seq]:
                    cached_pooled = self.cache[seq]["pooled"]
                    if cached_pooled.device != self.device:
                        cached_pooled = cached_pooled.to(self.device)
                    outputs.append(cached_pooled)
                    continue

                # Encode sequence
                emb = self._encode_sequence(seq, use_grad)
                pooled = self._pool_embedding(emb)

                # Cache pooled features (inference only)
                if not use_grad:
                    if seq not in self.cache:
                        self.cache[seq] = {}
                    self.cache[seq]["pooled"] = pooled.detach().cpu()

                outputs.append(pooled)

            return torch.stack(outputs, dim=0)

    def preload_cache(self, seq_list: List[str], device: torch.device = None):
        """
        Preload cache: Pre-encode common sequences to improve subsequent inference speed.

        Args:
            seq_list: List of sequences to preload
            device: Device (optional, defaults to encoder's device)
        """
        if device is None:
            device = self.device

        # Silent preload to avoid excessive output
        for seq in seq_list:
            if seq not in self.cache:
                try:
                    # Encode and cache pooled features
                    emb = self._encode_sequence(seq, use_grad=False)
                    pooled = self._pool_embedding(emb)

                    # Encode and cache token features
                    token_features = self._project_tokens(emb)

                    self.cache[seq] = {
                        "pooled": pooled.detach().cpu(),
                        "tokens": token_features.detach().cpu()
                    }
                except Exception:
                    # Silently handle preload failures
                    pass

    def clear_cache(self):
        """Clear cache"""
        self.cache.clear()

    def get_cache_info(self):
        """Get cache statistics"""
        total_sequences = len(self.cache)
        pooled_count = sum(1 for v in self.cache.values() if "pooled" in v)
        token_count = sum(1 for v in self.cache.values() if "tokens" in v)

        cache_memory = 0
        for seq_data in self.cache.values():
            for key, tensor in seq_data.items():
                cache_memory += tensor.numel() * tensor.element_size()

        return {
            "total_sequences": total_sequences,
            "pooled_features": pooled_count,
            "token_features": token_count,
            "cache_memory_mb": cache_memory / (1024 * 1024)
        }


class EnhancedSequenceEncoder(nn.Module):
    """Enhanced protein sequence encoder"""

    def __init__(self, device: torch.device = torch.device("cpu"),
                 proj_method: str = "conv",  # "conv", "mlp", "linear"
                 embed_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.device = device
        self.proj_method = proj_method
        self.embed_dim = embed_dim

        self.esmc = ESMC.from_pretrained("esmc_300m", device=device)
        self.tokenizer = EsmSequenceTokenizer()

        # Freeze strategy
        for p in self.esmc.parameters():
            p.requires_grad = False

        # Get ESMC output dimension
        with torch.no_grad():
            tmp_tokens = self.tokenizer.encode("M")
            tmp_tensor = ESMProteinTensor(sequence=torch.tensor(tmp_tokens).to(device))
            tmp_logits = self.esmc.logits(tmp_tensor, LogitsConfig(sequence=True, return_embeddings=True))
            self.d_model = tmp_logits.embeddings.shape[-1]

        # Advanced protein projector
        self.protein_projector = self._build_projector()

        # Cache
        self.cache = {}
        self.any_unfrozen = any(p.requires_grad for p in self.esmc.parameters())
        # Ensure module on target device
        self.to(self.device)

    def _build_projector(self):
        """Build protein feature projector"""
        if self.proj_method == "conv":
            return ConvPooler(self.d_model, self.embed_dim, kernel_size=3, stride=1)
        elif self.proj_method == "mlp":
            return nn.Sequential(
                nn.Linear(self.d_model, self.embed_dim * 4),
                nn.LayerNorm(self.embed_dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.embed_dim * 4, self.embed_dim * 2),
                nn.LayerNorm(self.embed_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                nn.LayerNorm(self.embed_dim)
            )
        else:
            return nn.Linear(self.d_model, self.embed_dim)

    def forward(self, seq_list: List[str]) -> torch.Tensor:
        """Build protein feature projector"""
        outputs = []

        for seq in seq_list:
            if not self.training and seq in self.cache:
                pooled = self.cache[seq].to(self.device)
                outputs.append(pooled)
                continue

            tokens = self.tokenizer.encode(seq)
            prot = ESMProteinTensor(sequence=torch.tensor(tokens).to(self.device))

            with torch.no_grad():
                logits_out = self.esmc.logits(prot, LogitsConfig(sequence=True, return_embeddings=True))
                emb = logits_out.embeddings

            # Ensure emb on correct device
            emb = emb.to(self.device)

            # Advanced dimensionality reduction
            if self.proj_method == "conv":
                # ConvPooler  [B, L, D]
                if emb.dim() == 2:
                    emb_in = emb.unsqueeze(0)  # [1, L, D]
                elif emb.dim() == 3 and emb.size(0) == 1:
                    emb_in = emb  # [1, L, D]
                else:
                    emb_in = emb  
                projected = self.protein_projector(emb_in)  # [1, L', embed_dim]
                pooled = projected.mean(dim=1).squeeze(0)  # [embed_dim]
            else:
                # MLP/Linear: [D]
                if emb.dim() == 2:  # [L, D]
                    emb_pooled = emb.mean(dim=0)  # [D]
                elif emb.dim() == 3:  # [B, L, D]
                    emb_pooled = emb.mean(dim=1).squeeze(0)  # [D]
                else:
                    emb_pooled = emb
                pooled = self.protein_projector(emb_pooled)  # [embed_dim]

            outputs.append(pooled)
            if not self.training:
                self.cache[seq] = pooled.detach().cpu()

        return torch.stack(outputs, dim=0)
