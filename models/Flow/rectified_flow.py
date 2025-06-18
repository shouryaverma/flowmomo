#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum
import numpy as np

from data.bioparse import VOCAB
from utils import register as R
from utils.oom_decorator import oom_decorator
from utils.nn_utils import SinusoidalTimeEmbeddings
from utils.gnn_utils import (
    length_to_batch_id, fully_connect_edges, knn_edges, 
    variadic_meshgrid, _edge_dist
)
from utils.chem_utils import (
    valence_check, cycle_check, sp2_check, check_stability,
    get_atom_valence, bond_to_valence, MAX_VALENCE, 
    connect_fragments, AtomVocab
)
from ..modules.create_net import create_net
from ..modules.nn import MLP, BlockEmbedding
from .flow_matching import FlowMatching


@R.register('MolecularRectifiedFlow')
class MolecularRectifiedFlow(nn.Module):
    def __init__(
        self,
        encoder_type: str = 'EPT',
        decoder_type: str = 'EPT', 
        hidden_size: int = 512,
        embed_size: int = 256,
        edge_size: int = 128,
        num_steps: int = 50,
        k_neighbors: int = 9,
        d_cutoff: float = 8.0,
        encoder_opt: dict = {},
        decoder_opt: dict = {},
        loss_weights: dict = {
            'coord_loss': 1.0,
            'atom_loss': 1.0, 
            'block_loss': 1.0,
            'bond_loss': 0.5,
            'constraint_loss': 0.1,
            'valence_loss': 0.2,
            'clash_loss': 0.15
        }
    ):
        super().__init__()
        
        # Use existing proven architectures
        self.encoder = create_net(encoder_type, hidden_size, edge_size, encoder_opt)
        self.decoder = create_net(decoder_type, hidden_size, edge_size, decoder_opt)
        
        # Flow matching core with improved discrete handling
        self.flow_matching = FlowMatching(num_steps)
        
        # Embeddings (reuse existing patterns)
        self.embedding = BlockEmbedding(VOCAB.get_num_block_type(), VOCAB.get_num_atom_type(), embed_size)
        self.edge_embedding = nn.Embedding(3, edge_size)  # [intra, inter, topo]
        self.time_embedding = SinusoidalTimeEmbeddings(hidden_size)
        
        # Input projection
        self.embed2hidden = nn.Linear(embed_size, hidden_size)
        self.time_proj = nn.Linear(hidden_size, hidden_size)
        
        # Flow velocity heads
        self.coord_velocity_head = nn.Linear(hidden_size, 3)
        self.atom_velocity_head = MLP(hidden_size, hidden_size, VOCAB.get_num_atom_type(), 2)
        self.block_velocity_head = MLP(hidden_size, hidden_size, VOCAB.get_num_block_type(), 2)
        
        self.k_neighbors = k_neighbors
        self.d_cutoff = d_cutoff
        self.loss_weights = loss_weights
        
        # Chemical constraint helpers
        self.atom_vocab = AtomVocab()

    @oom_decorator
    def forward(
        self,
        X,              # [Natom, 3] 
        S,              # [Nblock]
        A,              # [Natom]
        bonds,          # [Nbonds, 3]
        position_ids,   # [Nblock]
        chain_ids,      # [Nblock]
        generate_mask,  # [Nblock]
        block_lengths,  # [Nblock]
        lengths,        # [batch_size]
        is_aa,          # [Nblock]
        **kwargs
    ):
        # CORRECTED: Proper batch/block ID mapping
        batch_ids = length_to_batch_id(lengths)      # [Nblock] - batch ID for each block
        block_ids = length_to_batch_id(block_lengths) # [Natom] - block ID for each atom
        
        # Sample time and add noise (rectified flow interpolation)
        t, x_t, s_t, a_t, x_0, s_0, a_0 = self.flow_matching.add_noise(
            X, S, A, generate_mask, block_ids, batch_ids
        )
        
        # Prepare input features using existing embedding pattern
        H = self.embedding(s_t, a_t, block_ids)  # [Natom, embed_size]
        H = self.embed2hidden(H)  # [Natom, hidden_size]
        
        # CORRECTED: Time conditioning per atom
        atom_batch_ids = batch_ids[block_ids]  # [Natom] - batch ID for each atom
        max_batch_idx = len(t) - 1
        atom_batch_ids = torch.clamp(atom_batch_ids, 0, max_batch_idx)
        
        t_per_atom = t[atom_batch_ids]  # [Natom] - time for each atom
        t_embed = self.time_embedding(t_per_atom)  # [Natom, hidden_size]
        H = H + self.time_proj(t_embed)
        
        # CORRECTED: Get block-level edges (like original implementation)
        # batch_ids maps blocks to batches, chain_ids is per block
        edges, edge_type = self.get_edges(
            batch_ids, chain_ids, x_t, block_ids, generate_mask, 
            position_ids, is_aa, block_lengths
        )
        
        # Safety check for empty edges
        if edges.shape[1] == 0:
            # Create minimal edge between first two blocks if they exist
            if len(batch_ids) >= 2:
                edges = torch.tensor([[0], [1]], dtype=torch.long, device=X.device)
                edge_type = torch.zeros(1, dtype=torch.long, device=X.device)
            else:
                # Single block case - self edge
                edges = torch.tensor([[0], [0]], dtype=torch.long, device=X.device) 
                edge_type = torch.zeros(1, dtype=torch.long, device=X.device)
        
        edge_attr = self.edge_embedding(edge_type)
        
        # CORRECTED: EPT encoder expects these specific arguments
        # H: [Natom, hidden_size] - atom features
        # x_t: [Natom, 3] - atom coordinates  
        # block_ids: [Natom] - which block each atom belongs to
        # batch_ids: [Nblock] - which batch each block belongs to
        # edges: [2, E] - block-level edges
        # edge_attr: [E, edge_size] - edge features
        H_encoded, X_encoded = self.encoder(
            H, x_t, block_ids, batch_ids, edges, edge_attr
        )
        
        # Decode to get velocity predictions
        H_decoded, X_decoded = self.decoder(
            H_encoded, X_encoded, block_ids, batch_ids, edges, edge_attr
        )
        
        # Predict velocities
        v_x = self.coord_velocity_head(H_decoded)  # [Natom, 3]
        v_a = self.atom_velocity_head(H_decoded)   # [Natom, num_atom_types]
        
        # Block velocities (aggregate from atoms)
        v_s = self.block_velocity_head(H_decoded)  # [Natom, num_block_types]
        v_s = scatter_mean(v_s, block_ids, dim=0, dim_size=len(S))  # [Nblock, num_block_types]
        
        # Compute flow matching losses
        loss_dict = self.compute_flow_losses(
            v_x, v_a, v_s, X, A, S, x_0, s_0, a_0,
            generate_mask, block_ids, atom_batch_ids
        )
        
        # Add comprehensive chemical constraint losses
        constraint_losses = self.compute_chemical_constraints(
            x_t, a_t, s_t, generate_mask, block_ids, atom_batch_ids, 
            chain_ids, edges
        )
        loss_dict.update(constraint_losses)
        
        # Total loss
        total = sum(loss_dict[k] * self.loss_weights.get(k, 0) for k in loss_dict)
        loss_dict['total'] = total
        
        return loss_dict

    def get_edges(self, batch_ids, chain_ids, Z, block_ids, generate_mask, 
                position_ids, is_aa, block_lengths):
        """Create block-level edges (like original implementation)"""
        
        # CORRECTED: Create edges between BLOCKS, not atoms
        # batch_ids here should be batch IDs for each block
        # Z should be block-level coordinates (block centroids)
        
        # Get block-level coordinates (centroids of each block)
        from torch_scatter import scatter_mean
        
        # Convert atom coordinates to block centroids
        block_Z = scatter_mean(Z, block_ids, dim=0)  # [Nblock, 3]
        
        # Create all possible block-level edges within batches
        row, col = fully_connect_edges(batch_ids)  # batch_ids is [Nblock]
        
        if len(row) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=Z.device), \
                torch.empty(0, dtype=torch.long, device=Z.device)
        
        # Apply distance cutoff using block centroids
        distances = torch.norm(block_Z[row] - block_Z[col], dim=-1)
        close_mask = distances < self.d_cutoff
        row_filtered, col_filtered = row[close_mask], col[close_mask]
        
        if len(row_filtered) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=Z.device), \
                torch.empty(0, dtype=torch.long, device=Z.device)
        
        # Determine edge types based on block relationships
        edge_types = torch.zeros(len(row_filtered), dtype=torch.long, device=Z.device)
        
        # 1. Same chain edges (type 0 - intra)
        same_chain_mask = chain_ids[row_filtered] == chain_ids[col_filtered]
        edge_types[same_chain_mask] = 0
        
        # 2. Different chain edges (type 1 - inter) 
        diff_chain_mask = chain_ids[row_filtered] != chain_ids[col_filtered]
        edge_types[diff_chain_mask] = 1
        
        # 3. Sequential edges (type 2 - topo) - adjacent in sequence
        sequential_mask = (torch.abs(position_ids[row_filtered] - position_ids[col_filtered]) == 1) & same_chain_mask
        edge_types[sequential_mask] = 2
        
        edges = torch.stack([row_filtered, col_filtered])
        return edges, edge_types

    def compute_flow_losses(self, v_x, v_a, v_s, X_true, A_true, S_true, 
                        x_0, s_0, a_0, generate_mask, block_ids, batch_ids):
        loss_dict = {}
        
        # CORRECT: True velocity fields v = x_1 - x_0
        target_v_x = X_true - x_0
        mask_atoms = generate_mask[block_ids]
        
        if mask_atoms.any():
            loss_dict['coord_loss'] = F.mse_loss(v_x[mask_atoms], target_v_x[mask_atoms])
            loss_dict['atom_loss'] = F.cross_entropy(v_a[mask_atoms], A_true[mask_atoms])
        else:
            loss_dict['coord_loss'] = torch.tensor(0.0, device=X_true.device)
            loss_dict['atom_loss'] = torch.tensor(0.0, device=X_true.device)
        
        if generate_mask.any():
            loss_dict['block_loss'] = F.cross_entropy(v_s[generate_mask], S_true[generate_mask])
        else:
            loss_dict['block_loss'] = torch.tensor(0.0, device=X_true.device)
            
        return loss_dict

    def compute_chemical_constraints(self, x_pred, a_pred, s_pred, generate_mask, 
                                   block_ids, batch_ids, chain_ids, edges):
        """Comprehensive chemical constraints from original implementation"""
        constraint_losses = {}
        device = x_pred.device
        
        # Initialize all losses
        constraint_losses['constraint_loss'] = torch.tensor(0.0, device=device)
        constraint_losses['valence_loss'] = torch.tensor(0.0, device=device)
        constraint_losses['clash_loss'] = torch.tensor(0.0, device=device)
        
        # Only apply to generated atoms
        gen_mask_atoms = generate_mask[block_ids]
        if not gen_mask_atoms.any():
            return constraint_losses
        
        try:
            # 1. Valence constraint loss
            valence_loss = self.compute_valence_loss(x_pred, a_pred, gen_mask_atoms, edges)
            constraint_losses['valence_loss'] = valence_loss
            
            # 2. Clash avoidance loss 
            clash_loss = self.compute_clash_loss(x_pred, a_pred, gen_mask_atoms, batch_ids)
            constraint_losses['clash_loss'] = clash_loss
            
            # 3. Bond length guidance
            bond_loss = self.compute_bond_length_guidance(x_pred, a_pred, gen_mask_atoms, edges)
            constraint_losses['constraint_loss'] = bond_loss
            
        except Exception as e:
            # If constraint computation fails, continue with zero losses
            print(f"Warning: Chemical constraint computation failed: {e}")
        
        return constraint_losses

    def compute_valence_loss(self, x_pred, a_pred, gen_mask, edges):
        """Compute valence constraint violation penalty"""
        if not gen_mask.any() or edges.shape[1] == 0:
            return torch.tensor(0.0, device=x_pred.device)
        
        valence_penalty = 0.0
        gen_indices = torch.where(gen_mask)[0]
        
        # Get predicted atom types
        pred_atoms = torch.argmax(a_pred[gen_mask], dim=-1)
        
        # Check valence for generated atoms
        for i, atom_idx in enumerate(gen_indices):
            try:
                # Get atom type (convert to symbol)
                atom_type_idx = pred_atoms[i].item()
                if atom_type_idx < len(self.atom_vocab.idx2atom):
                    atom_symbol = self.atom_vocab.idx2atom[atom_type_idx]
                    
                    if atom_symbol in MAX_VALENCE:
                        # Count bonds for this atom
                        edge_mask = (edges[0] == atom_idx) | (edges[1] == atom_idx)
                        num_bonds = edge_mask.sum().float()
                        max_val = MAX_VALENCE[atom_symbol]
                        
                        # Penalty for exceeding valence
                        if num_bonds > max_val:
                            valence_penalty += (num_bonds - max_val) ** 2
                            
            except Exception:
                continue
        
        return torch.tensor(valence_penalty / (len(gen_indices) + 1e-8), device=x_pred.device)

    def compute_clash_loss(self, x_pred, a_pred, gen_mask, batch_ids):
        """Compute atomic clash penalty"""
        if not gen_mask.any():
            return torch.tensor(0.0, device=x_pred.device)
        
        clash_penalty = 0.0
        gen_coords = x_pred[gen_mask]
        
        # Van der Waals radii (simplified)
        vdw_radii = {'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'P': 1.8, 'S': 1.8}
        
        if len(gen_coords) > 1:
            # Pairwise distances
            dist_matrix = torch.cdist(gen_coords, gen_coords)
            
            # Get atom types for generated atoms
            pred_atoms = torch.argmax(a_pred[gen_mask], dim=-1)
            
            # Check for clashes
            for i in range(len(gen_coords)):
                for j in range(i + 1, len(gen_coords)):
                    try:
                        atom_i = self.atom_vocab.idx2atom[pred_atoms[i].item()]
                        atom_j = self.atom_vocab.idx2atom[pred_atoms[j].item()]
                        
                        if atom_i in vdw_radii and atom_j in vdw_radii:
                            min_dist = vdw_radii[atom_i] + vdw_radii[atom_j]
                            actual_dist = dist_matrix[i, j]
                            
                            if actual_dist < min_dist:
                                clash_penalty += (min_dist - actual_dist) ** 2
                    except Exception:
                        continue

        # return torch.tensor(clash_penalty / (gen_mask.sum() + 1e-8), device=x_pred.device)
        return clash_penalty / (gen_mask.sum() + 1e-8)

    def compute_bond_length_guidance(self, x_pred, a_pred, gen_mask, edges):
        """Guide bond lengths toward chemically reasonable values"""
        if not gen_mask.any() or edges.shape[1] == 0:
            return torch.tensor(0.0, device=x_pred.device)
        
        bond_penalty = 0.0
        
        # Typical bond lengths (in Angstroms)
        bond_lengths = {
            ('C', 'C'): 1.54, ('C', 'N'): 1.47, ('C', 'O'): 1.43,
            ('N', 'N'): 1.45, ('N', 'O'): 1.40, ('O', 'O'): 1.48,
            ('C', 'H'): 1.09, ('N', 'H'): 1.01, ('O', 'H'): 0.96
        }
        
        pred_atoms = torch.argmax(a_pred, dim=-1)
        
        for edge_idx in range(edges.shape[1]):
            atom1_idx, atom2_idx = edges[0, edge_idx], edges[1, edge_idx]
            
            # Only penalize if at least one atom is generated
            if gen_mask[atom1_idx] or gen_mask[atom2_idx]:
                try:
                    atom1_type = self.atom_vocab.idx2atom[pred_atoms[atom1_idx].item()]
                    atom2_type = self.atom_vocab.idx2atom[pred_atoms[atom2_idx].item()]
                    
                    # Get expected bond length
                    bond_key = tuple(sorted([atom1_type, atom2_type]))
                    if bond_key in bond_lengths:
                        expected_length = bond_lengths[bond_key]
                        actual_length = torch.norm(x_pred[atom1_idx] - x_pred[atom2_idx])
                        
                        # Soft constraint - penalize deviations
                        bond_penalty += ((actual_length - expected_length) ** 2)
                        
                except Exception:
                    continue
        
        return torch.tensor(bond_penalty / (edges.shape[1] + 1e-8), device=x_pred.device)

    @torch.no_grad()
    def sample(
        self,
        X, S, A, bonds, position_ids, chain_ids, generate_mask, 
        block_lengths, lengths, is_aa, num_steps=50, **kwargs
    ):
        """Sample using rectified flow with chemical constraints"""
        return self.flow_matching.sample(
            self, X, S, A, bonds, position_ids, chain_ids, generate_mask,
            block_lengths, lengths, is_aa, num_steps, **kwargs
        )

    def generate(self, *args, **kwargs):
        """Alias for sample to match existing interface"""
        return self.sample(*args, **kwargs)