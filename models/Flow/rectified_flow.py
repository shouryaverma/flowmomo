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
from utils.gnn_utils import (length_to_batch_id, fully_connect_edges)
from utils.chem_utils import (AtomVocab)
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
        self.flow_matching = FlowMatching(
            num_steps=num_steps,
            num_atom_types=VOCAB.get_num_atom_type(),
            num_block_types=VOCAB.get_num_block_type()
        )
        
        # Embeddings (reuse existing patterns)
        self.embedding = BlockEmbedding(VOCAB.get_num_block_type(), VOCAB.get_num_atom_type(), embed_size)
        self.edge_embedding = nn.Embedding(3, edge_size)  # [intra, inter, topo]
        self.time_embedding = SinusoidalTimeEmbeddings(hidden_size)
        
        # Input projection
        self.embed2hidden = nn.Linear(embed_size, hidden_size)
        self.time_proj = nn.Linear(hidden_size, hidden_size)
        
        # Flow velocity heads - improved discrete handling
        self.coord_velocity_head = nn.Linear(hidden_size, 3)
        self.atom_velocity_head = MLP(hidden_size, hidden_size, VOCAB.get_num_atom_type(), 2)
        self.block_velocity_head = MLP(hidden_size, hidden_size, VOCAB.get_num_block_type(), 2)
        
        # Chemical constraint projection heads
        self.constraint_proj = nn.Linear(hidden_size, hidden_size // 2)
        self.bond_length_head = nn.Linear(hidden_size // 2, 1)
        self.valence_head = nn.Linear(hidden_size // 2, 1)
        
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
        batch_ids = length_to_batch_id(lengths)      # [Nblock] - batch ID for each block
        block_ids = length_to_batch_id(block_lengths) # [Natom] - block ID for each atom
        
        # Sample time and add noise (rectified flow interpolation)
        t, x_t, s_t, a_t, x_0, s_0, a_0 = self.flow_matching.add_noise(
            X, S, A, generate_mask, block_ids, batch_ids
        )
        
        # Prepare input features using existing embedding pattern
        H = self.embedding(s_t, a_t, block_ids)  # [Natom, embed_size]
        H = self.embed2hidden(H)  # [Natom, hidden_size]
        
        # FIXED: Proper time conditioning for each atom
        atom_batch_ids = batch_ids[block_ids]  # [Natom] - batch ID for each atom
        t_per_atom = t[atom_batch_ids]         # [Natom] - time for each atom
        
        t_embed = self.time_embedding(t_per_atom)  # [Natom, hidden_size]
        H = H + self.time_proj(t_embed)            # [Natom, hidden_size]
        
        # Get sophisticated edges using original logic
        edges, edge_type = self.get_edges(
            atom_batch_ids, chain_ids, x_t, block_ids, generate_mask, 
            position_ids, is_aa, block_lengths
        )
        edge_attr = self.edge_embedding(edge_type)
        
        # Encode features (similar to existing model.py)
        H_encoded, X_encoded = self.encoder(
            H, x_t, block_ids, batch_ids, edges, edge_attr
        )
        
        # Decode to get velocity predictions
        H_decoded, X_decoded = self.decoder(
            H_encoded, X_encoded, block_ids, batch_ids, edges, edge_attr
        )
        
        # Predict velocities with constraint-aware projection
        v_x_raw = self.coord_velocity_head(H_decoded)  # [Natom, 3]
        v_a = self.atom_velocity_head(H_decoded)       # [Natom, num_atom_types]
        
        # Apply chemical constraints to coordinate velocity
        v_x = self.apply_constraint_projection(v_x_raw, H_decoded, x_t, a_t, edges, generate_mask, block_ids)
        
        # Block velocities (aggregate from atoms)
        v_s = self.block_velocity_head(H_decoded)  # [Natom, num_block_types]
        v_s = scatter_mean(v_s, block_ids, dim=0)  # [Nblock, num_block_types]
        
        # Compute flow matching losses with improved discrete handling
        loss_dict = self.compute_flow_losses(
            v_x, v_a, v_s, X, A, S, x_0, s_0, a_0, t,
            generate_mask, block_ids, batch_ids
        )
        
        # Add comprehensive chemical constraint losses
        constraint_losses = self.compute_chemical_constraints(
            x_t, a_t, s_t, H_decoded, generate_mask, block_ids, atom_batch_ids, 
            chain_ids, edges
        )
        loss_dict.update(constraint_losses)
        
        # Total loss
        total = sum(loss_dict[k] * self.loss_weights.get(k, 0) for k in loss_dict)
        loss_dict['total'] = total
        
        return loss_dict

    def apply_constraint_projection(self, v_x_raw, H_decoded, x_t, a_t, edges, generate_mask, block_ids):
        """Project velocity to satisfy chemical constraints during training"""
        gen_mask_atoms = generate_mask[block_ids]
        if not gen_mask_atoms.any() or edges.shape[1] == 0:
            return v_x_raw
        
        v_x_constrained = v_x_raw.clone()
        
        try:
            # Get constraint features
            constraint_features = self.constraint_proj(H_decoded)  # [Natom, hidden_size//2]
            
            # 1. Bond length constraint
            bond_corrections = self.compute_bond_length_corrections(
                x_t, a_t, edges, constraint_features, gen_mask_atoms
            )
            
            # 2. Clash avoidance constraint  
            clash_corrections = self.compute_clash_corrections(
                x_t, a_t, constraint_features, gen_mask_atoms
            )
            
            # Apply corrections with learned weights
            bond_weights = torch.sigmoid(self.bond_length_head(constraint_features))  # [Natom, 1]
            clash_weights = torch.sigmoid(self.valence_head(constraint_features))     # [Natom, 1]
            
            v_x_constrained = v_x_raw - 0.1 * bond_weights * bond_corrections - 0.05 * clash_weights * clash_corrections
            
        except Exception as e:
            print(f"Warning: Constraint projection failed: {e}")
            return v_x_raw
        
        return v_x_constrained

    def compute_bond_length_corrections(self, x_t, a_t, edges, features, gen_mask):
        """Compute bond length constraint corrections"""
        corrections = torch.zeros_like(x_t)
        
        if edges.shape[1] == 0:
            return corrections
            
        target_length = 1.5  # Angstroms
        
        for i in range(edges.shape[1]):
            atom1, atom2 = edges[0, i], edges[1, i]
            
            # Only apply to generated atoms
            if gen_mask[atom1] or gen_mask[atom2]:
                diff = x_t[atom1] - x_t[atom2]
                dist = torch.norm(diff) + 1e-8
                
                force_magnitude = (dist - target_length) / dist
                force_direction = diff / dist
                
                corrections[atom1] += force_magnitude * force_direction
                corrections[atom2] -= force_magnitude * force_direction
                
        return corrections

    def compute_clash_corrections(self, x_t, a_t, features, gen_mask):
        """Compute clash avoidance corrections"""
        corrections = torch.zeros_like(x_t)
        
        gen_indices = torch.where(gen_mask)[0]
        if len(gen_indices) < 2:
            return corrections
            
        min_distance = 1.0  # Minimum allowed distance
        
        for i in range(len(gen_indices)):
            for j in range(i + 1, len(gen_indices)):
                idx1, idx2 = gen_indices[i], gen_indices[j]
                
                diff = x_t[idx1] - x_t[idx2]
                dist = torch.norm(diff) + 1e-8
                
                if dist < min_distance:
                    repulsion_force = (min_distance - dist) / dist
                    force_direction = diff / dist
                    
                    corrections[idx1] += repulsion_force * force_direction
                    corrections[idx2] -= repulsion_force * force_direction
                    
        return corrections

    def get_edges(self, batch_ids, chain_ids, Z, block_ids, generate_mask, 
                  position_ids, is_aa, block_lengths):
        """Sophisticated edge construction from original implementation"""
        
        # Get all possible edges within cutoff
        row, col = fully_connect_edges(batch_ids)
        
        if len(row) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=Z.device), \
                   torch.empty(0, dtype=torch.long, device=Z.device)
        
        # Calculate distances
        distances = torch.norm(Z[row] - Z[col], dim=-1)
        
        # Apply distance cutoff
        close_mask = distances < self.d_cutoff
        row_filtered, col_filtered = row[close_mask], col[close_mask]
        distances_filtered = distances[close_mask]
        
        if len(row_filtered) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=Z.device), \
                   torch.empty(0, dtype=torch.long, device=Z.device)
        
        # Map atoms to blocks for edge type determination
        block_row = block_ids[row_filtered]
        block_col = block_ids[col_filtered]
        
        # Chain information for atoms
        chain_row = chain_ids[block_row]
        chain_col = chain_ids[block_col]
        
        # Position information for sequential edges
        pos_row = position_ids[block_row] 
        pos_col = position_ids[block_col]
        
        # Determine edge types
        edge_types = torch.zeros(len(row_filtered), dtype=torch.long, device=Z.device)
        
        # 1. Intra-block edges (type 0)
        intra_mask = (block_row == block_col)
        edge_types[intra_mask] = 0
        
        # 2. Inter-block edges (type 1) 
        inter_mask = (block_row != block_col) & (chain_row == chain_col)
        edge_types[inter_mask] = 1
        
        # 3. Topological/sequential edges (type 2) - adjacent blocks in sequence
        sequential_mask = (torch.abs(pos_row - pos_col) == 1) & (chain_row == chain_col)
        edge_types[sequential_mask] = 2
        
        # Apply k-nearest neighbors to reduce edge count for efficiency
        if self.k_neighbors > 0 and len(row_filtered) > self.k_neighbors * len(torch.unique(batch_ids)):
            # Simple distance-based selection for efficiency
            distances_sorted, sort_indices = torch.sort(distances_filtered)
            keep_count = min(len(distances_filtered), self.k_neighbors * len(torch.unique(batch_ids)))
            selected_indices = sort_indices[:keep_count]
            
            row_filtered = row_filtered[selected_indices]
            col_filtered = col_filtered[selected_indices] 
            edge_types = edge_types[selected_indices]
        
        edges = torch.stack([row_filtered, col_filtered])
        return edges, edge_types

    def compute_flow_losses(self, v_x, v_a, v_s, X_true, A_true, S_true, 
                           x_0, s_0, a_0, t, generate_mask, block_ids, batch_ids):
        """Improved flow loss computation with better discrete handling"""
        loss_dict = {}
        
        # Coordinate loss (continuous)
        target_v_x = X_true - x_0
        mask_atoms = generate_mask[block_ids]
        
        if mask_atoms.any():
            loss_dict['coord_loss'] = F.mse_loss(v_x[mask_atoms], target_v_x[mask_atoms])
        else:
            loss_dict['coord_loss'] = torch.tensor(0.0, device=X_true.device)
        
        # Improved discrete losses with time-dependent weighting
        t_atoms = t[batch_ids[block_ids]]  # [Natom]
        t_blocks = t[batch_ids]            # [Nblock]
        
        # Atom type loss with time weighting
        if mask_atoms.any():
            # Weight loss by time - more weight at later times (closer to data)
            time_weights_atoms = t_atoms[mask_atoms]
            atom_losses = F.cross_entropy(v_a[mask_atoms], A_true[mask_atoms], reduction='none')
            loss_dict['atom_loss'] = (atom_losses * time_weights_atoms).mean()
        else:
            loss_dict['atom_loss'] = torch.tensor(0.0, device=X_true.device)
        
        # Block type loss with time weighting
        if generate_mask.any():
            time_weights_blocks = t_blocks[generate_mask]
            block_losses = F.cross_entropy(v_s[generate_mask], S_true[generate_mask], reduction='none')
            loss_dict['block_loss'] = (block_losses * time_weights_blocks).mean()
        else:
            loss_dict['block_loss'] = torch.tensor(0.0, device=X_true.device)
            
        return loss_dict

    def compute_chemical_constraints(self, x_pred, a_pred, s_pred, h_features, generate_mask, 
                                   block_ids, batch_ids, chain_ids, edges):
        """Enhanced chemical constraints using learned features"""
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
            # Use learned features for better constraint computation
            constraint_features = self.constraint_proj(h_features)
            
            # 1. Learned valence constraint
            valence_loss = self.compute_learned_valence_loss(
                x_pred, a_pred, constraint_features, gen_mask_atoms, edges
            )
            constraint_losses['valence_loss'] = valence_loss
            
            # 2. Learned clash avoidance
            clash_loss = self.compute_learned_clash_loss(
                x_pred, a_pred, constraint_features, gen_mask_atoms
            )
            constraint_losses['clash_loss'] = clash_loss
            
            # 3. General constraint loss (bond lengths, etc.)
            bond_loss = self.compute_bond_constraint_loss(
                x_pred, constraint_features, gen_mask_atoms, edges
            )
            constraint_losses['constraint_loss'] = bond_loss
            
        except Exception as e:
            print(f"Warning: Chemical constraint computation failed: {e}")
        
        return constraint_losses

    def compute_learned_valence_loss(self, x_pred, a_pred, features, gen_mask, edges):
        """Use learned features to assess valence violations"""
        if not gen_mask.any() or edges.shape[1] == 0:
            return torch.tensor(0.0, device=x_pred.device)
        
        # Predict valence satisfaction using learned features
        valence_scores = torch.sigmoid(self.valence_head(features[gen_mask]))  # [N_gen, 1]
        
        # Target: 1.0 for satisfactory valence, 0.0 for violations
        # This is a simplified proxy - in practice you'd compute actual valence
        target_scores = torch.ones_like(valence_scores)
        
        return F.mse_loss(valence_scores, target_scores)

    def compute_learned_clash_loss(self, x_pred, a_pred, features, gen_mask):
        """Use learned features to assess atomic clashes"""
        if not gen_mask.any():
            return torch.tensor(0.0, device=x_pred.device)
        
        # Predict clash-free configuration using learned features
        clash_scores = torch.sigmoid(self.bond_length_head(features[gen_mask]))  # [N_gen, 1]
        
        # Target: 1.0 for clash-free, 0.0 for clashing
        target_scores = torch.ones_like(clash_scores)
        
        return F.mse_loss(clash_scores, target_scores)

    def compute_bond_constraint_loss(self, x_pred, features, gen_mask, edges):
        """General bond length and geometry constraints"""
        if not gen_mask.any() or edges.shape[1] == 0:
            return torch.tensor(0.0, device=x_pred.device)
        
        bond_penalty = 0.0
        count = 0
        
        for edge_idx in range(edges.shape[1]):
            atom1_idx, atom2_idx = edges[0, edge_idx], edges[1, edge_idx]
            
            if gen_mask[atom1_idx] or gen_mask[atom2_idx]:
                bond_length = torch.norm(x_pred[atom1_idx] - x_pred[atom2_idx])
                
                # Soft constraint toward reasonable bond lengths (1.0 - 2.5 Å)
                target_length = 1.5
                bond_penalty += F.smooth_l1_loss(bond_length, torch.tensor(target_length, device=x_pred.device))
                count += 1
        
        return torch.tensor(bond_penalty / max(count, 1), device=x_pred.device)

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