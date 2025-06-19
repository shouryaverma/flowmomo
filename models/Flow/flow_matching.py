import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
import numpy as np

from utils.gnn_utils import length_to_batch_id
from utils.chem_utils import (AtomVocab)


class FlowMatching(nn.Module):
    def __init__(self, num_steps=50, sigma_min=1e-4, temp_schedule='cosine', 
                 num_atom_types=None, num_block_types=None):
        super().__init__()
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.temp_schedule = temp_schedule
        self.atom_vocab = AtomVocab()
        
        # For improved discrete handling
        self.num_atom_types = num_atom_types or 32
        self.num_block_types = num_block_types or 64
        
        # Discrete noise schedules
        self.register_buffer('alpha_schedule', self._get_alpha_schedule(num_steps))
        
    def _get_alpha_schedule(self, num_steps):
        """Get noise schedule for discrete variables"""
        if self.temp_schedule == 'cosine':
            t = torch.linspace(0, 1, num_steps)
            alpha = torch.cos(t * np.pi / 2) ** 2
        elif self.temp_schedule == 'linear':
            alpha = torch.linspace(1.0, 0.1, num_steps)
        else:
            alpha = torch.ones(num_steps)
        return alpha
        
    def add_noise(self, X, S, A, generate_mask, block_ids, batch_ids, t=None):
        """Improved noise addition with better discrete handling"""
        batch_size = batch_ids.max() + 1
        
        if t is None:
            t = torch.rand(batch_size, device=X.device)
        
        # Sample from prior (x_0) - more chemically informed
        x_0 = self.sample_chemical_prior(X, A, generate_mask, block_ids)
        s_0 = self.sample_discrete_prior(S, generate_mask, self.num_block_types)
        a_0 = self.sample_discrete_prior(A, generate_mask[block_ids], self.num_atom_types)
        
        # Linear interpolation: x_t = (1-t) * x_0 + t * x_1
        t_expand_atoms = t[batch_ids[block_ids]]
        t_expand_blocks = t[batch_ids]
        
        x_t = (1 - t_expand_atoms[..., None]) * x_0 + t_expand_atoms[..., None] * X
        
        # Improved discrete interpolation with corruption
        s_t = self.improved_discrete_interpolation(s_0, S, t_expand_blocks, self.num_block_types)
        a_t = self.improved_discrete_interpolation(a_0, A, t_expand_atoms, self.num_atom_types)
        
        # Only interpolate in generation regions
        gen_mask_atoms = generate_mask[block_ids]
        x_t = torch.where(gen_mask_atoms[..., None], x_t, X)
        s_t = torch.where(generate_mask, s_t, S)
        a_t = torch.where(gen_mask_atoms, a_t, A)
        
        return t, x_t, s_t, a_t, x_0, s_0, a_0

    def sample_chemical_prior(self, X, A, generate_mask, block_ids):
        """Enhanced chemically-informed coordinate prior"""
        x_0 = torch.randn_like(X)
        
        gen_mask_atoms = generate_mask[block_ids]
        if gen_mask_atoms.any():
            ctx_mask = ~gen_mask_atoms
            if ctx_mask.any():
                ctx_coords = X[ctx_mask]
                
                if len(ctx_coords) > 0:
                    # More sophisticated placement strategy
                    gen_indices = torch.where(gen_mask_atoms)[0]
                    
                    for i, atom_idx in enumerate(gen_indices):
                        # Find 1-3 nearest context atoms
                        distances = torch.norm(ctx_coords - X[atom_idx], dim=-1)
                        nearest_indices = torch.topk(distances, min(3, len(ctx_coords)), largest=False)[1]
                        nearest_coords = ctx_coords[nearest_indices]
                        
                        if len(nearest_coords) == 1:
                            # Place at reasonable distance from single neighbor
                            direction = torch.randn(3, device=X.device)
                            direction = direction / (torch.norm(direction) + 1e-8)
                            distance = 1.5 + torch.rand(1, device=X.device) * 1.0  # 1.5-2.5 Å
                            x_0[atom_idx] = nearest_coords[0] + direction * distance
                        else:
                            # Place as weighted average with noise
                            weights = 1.0 / (distances[nearest_indices] + 1e-8)
                            weights = weights / weights.sum()
                            centroid = (nearest_coords * weights[:, None]).sum(dim=0)
                            noise = torch.randn(3, device=X.device) * 0.5
                            x_0[atom_idx] = centroid + noise
                else:
                    # Fallback: use context centroid
                    centroid = ctx_coords.mean(dim=0)
                    noise_scale = 2.0
                    x_0[gen_mask_atoms] = centroid[None, :] + noise_scale * torch.randn_like(x_0[gen_mask_atoms])
        
        return x_0

    def sample_discrete_prior(self, true_vals, mask, num_classes):
        """Improved discrete variable prior"""
        if mask.any():
            # Use uniform distribution over valid range
            max_val = min(true_vals.max().item() + 1, num_classes)
            prior_samples = torch.randint(0, max_val, true_vals.shape, device=true_vals.device)
            return torch.where(mask, prior_samples, true_vals)
        return true_vals
        
    def improved_discrete_interpolation(self, x_0, x_1, t, num_classes):
        """Better discrete variable interpolation with corruption"""
        # Clamp values to valid range
        x_0 = x_0.clamp(0, num_classes - 1)
        x_1 = x_1.clamp(0, num_classes - 1)
        
        # Use corruption-based approach instead of pure interpolation
        # At t=0, we have pure noise; at t=1, we have pure data
        corruption_prob = 1.0 - t  # High corruption at low t
        
        # Create mask for corruption
        corrupt_mask = torch.rand_like(t) < corruption_prob
        
        # For corrupted positions, use uniform noise
        uniform_samples = torch.randint(0, num_classes, x_1.shape, device=x_1.device)
        
        # Progressive transition from noise to data
        # Use gumbel-softmax for smooth interpolation
        x_0_oh = F.one_hot(x_0, num_classes).float()
        x_1_oh = F.one_hot(x_1, num_classes).float()
        
        # Temperature decreases as we approach data (t approaches 1)
        temperature = 0.1 + 0.9 * (1 - t)
        
        # Interpolate in probability space
        x_t_logits = (1 - t[..., None]) * x_0_oh + t[..., None] * x_1_oh
        x_t_logits = x_t_logits / (temperature[..., None] + 1e-8)
        
        # Add gumbel noise for sampling
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(x_t_logits) + 1e-8) + 1e-8)
        x_t_logits = x_t_logits + 0.1 * gumbel_noise
        
        # Sample from the distribution
        x_t = torch.argmax(x_t_logits, dim=-1)
        
        # Apply corruption mask
        x_t = torch.where(corrupt_mask, uniform_samples, x_t)
        
        return x_t.clamp(0, num_classes - 1)
        
    @torch.no_grad()
    def sample(self, model, X, S, A, bonds, position_ids, chain_ids, 
               generate_mask, block_lengths, lengths, is_aa, num_steps=50, 
               apply_constraints=True, **kwargs):
        """Fixed Euler sampling for rectified flow"""
        
        batch_ids = length_to_batch_id(lengths)
        block_ids = length_to_batch_id(block_lengths)
        
        # Initialize from chemically-informed noise in generation regions
        x_t = self.sample_chemical_prior(X, A, generate_mask, block_ids)
        x_t = torch.where(generate_mask[block_ids][..., None], x_t, X)
        
        s_t = self.sample_discrete_prior(S, generate_mask, self.num_block_types)
        a_t = self.sample_discrete_prior(A, generate_mask[block_ids], self.num_atom_types)
        
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((batch_ids.max() + 1,), i * dt, device=X.device)
            
            # Get current temperature for discrete updates
            temp = self.get_temperature(i, num_steps)
            
            # Forward pass through model
            H = model.embedding(s_t, a_t, block_ids)
            H = model.embed2hidden(H)
            
            # FIXED: Proper time conditioning
            atom_batch_ids = batch_ids[block_ids]  # [Natom] - batch ID for each atom
            t_embed = model.time_embedding(t[atom_batch_ids])  # Use atom_batch_ids, not batch_ids
            H = H + model.time_proj(t_embed)
            
            # Get edges for current state
            edges, edge_type = model.get_edges(
                atom_batch_ids, chain_ids, x_t, block_ids, generate_mask, 
                position_ids, is_aa, block_lengths
            )
            
            # Handle empty edges case
            if edges.shape[1] > 0:
                edge_attr = model.edge_embedding(edge_type)
            else:
                edge_attr = torch.empty((0, model.edge_embedding.embedding_dim), device=X.device)
            
            # Encode then decode
            H_encoded, X_encoded = model.encoder(H, x_t, block_ids, atom_batch_ids, edges, edge_attr)
            H_decoded, X_decoded = model.decoder(H_encoded, X_encoded, block_ids, atom_batch_ids, edges, edge_attr)
            
            # Get velocity predictions
            v_x = model.coord_velocity_head(H_decoded)
            v_a = model.atom_velocity_head(H_decoded)
            v_s = model.block_velocity_head(H_decoded)
            v_s = scatter_mean(v_s, block_ids, dim=0, dim_size=s_t.shape[0])
            
            # Update using Euler method
            x_t_new = x_t + v_x * dt
            s_t_new = self.update_discrete_improved(s_t, v_s, dt, generate_mask, temp, self.num_block_types)
            a_t_new = self.update_discrete_improved(a_t, v_a, dt, generate_mask[block_ids], temp, self.num_atom_types)
            
            # Apply chemical constraints if enabled
            if apply_constraints:
                x_t_new, a_t_new = self.apply_chemical_constraints(
                    x_t_new, a_t_new, s_t_new, generate_mask, block_ids, 
                    atom_batch_ids, chain_ids, edges
                )
            
            # Update states (only in generation regions)
            x_t = torch.where(generate_mask[block_ids][..., None], x_t_new, x_t)
            s_t = torch.where(generate_mask, s_t_new, s_t)
            a_t = torch.where(generate_mask[block_ids], a_t_new, a_t)
            
        return self.format_output(x_t, s_t, a_t, generate_mask, block_ids, lengths)

    def get_temperature(self, step, total_steps):
        """Improved temperature schedule"""
        progress = step / total_steps
        
        if self.temp_schedule == 'cosine':
            return 0.05 + 0.95 * (1 + np.cos(np.pi * progress)) / 2
        elif self.temp_schedule == 'linear':
            return 1.0 - 0.95 * progress
        else:
            return max(0.1, 1.0 - 0.9 * progress)
    
    def update_discrete_improved(self, x_t, v_logits, dt, mask, temperature, num_classes):
        """Improved discrete variable updates"""
        if not mask.any():
            return x_t
            
        # Clamp current values to valid range
        x_t = x_t.clamp(0, num_classes - 1)
        
        # Apply temperature to logits
        v_logits_temp = v_logits / (temperature + 1e-8)
        probs = F.softmax(v_logits_temp, dim=-1)
        
        # Current state as one-hot
        x_t_oh = F.one_hot(x_t, num_classes).float()
        
        # Gumbel-Softmax for smoother updates
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(probs) + 1e-8) + 1e-8)
        
        # Interpolate toward velocity direction with Gumbel noise
        x_t_oh_new = x_t_oh + dt * (probs - x_t_oh) + 0.01 * dt * gumbel_noise
        x_t_oh_new = F.softmax(x_t_oh_new / temperature, dim=-1)
        
        # Sample new discrete values
        x_new = torch.multinomial(x_t_oh_new + 1e-8, 1).squeeze(-1)
        x_new = x_new.clamp(0, num_classes - 1)
        
        # Only update in mask regions
        return torch.where(mask, x_new, x_t)

    def apply_chemical_constraints(self, x_pred, a_pred, s_pred, generate_mask, 
                                 block_ids, batch_ids, chain_ids, edges):
        """Enhanced chemical constraints during sampling"""
        gen_mask_atoms = generate_mask[block_ids]
        if not gen_mask_atoms.any():
            return x_pred, a_pred
        
        x_corrected = x_pred.clone()
        a_corrected = a_pred.clone()
        
        try:
            # 1. Avoid atomic clashes with adaptive distance
            x_corrected = self.avoid_clashes_adaptive(x_corrected, a_corrected, gen_mask_atoms)
            
            # 2. Enforce reasonable bond lengths
            if edges.shape[1] > 0:
                x_corrected = self.enforce_bond_lengths_adaptive(x_corrected, a_corrected, edges, gen_mask_atoms)
            
            # 3. Fix valence violations with chemical knowledge
            a_corrected = self.fix_valence_violations_smart(x_corrected, a_corrected, gen_mask_atoms, edges)
            
        except Exception as e:
            print(f"Warning: Constraint application failed: {e}")
            return x_pred, a_pred
        
        return x_corrected, a_corrected

    def avoid_clashes_adaptive(self, x_pred, a_pred, gen_mask):
        """Adaptive clash avoidance based on atom types"""
        if not gen_mask.any():
            return x_pred
            
        x_corrected = x_pred.clone()
        gen_coords = x_corrected[gen_mask]
        
        if len(gen_coords) <= 1:
            return x_corrected
            
        # Get predicted atom types
        pred_atom_types = torch.argmax(a_pred[gen_mask], dim=-1)
        
        # Van der Waals radii
        vdw_radii = {0: 1.2, 1: 1.7, 2: 1.55, 3: 1.52, 4: 1.47, 5: 1.8, 6: 1.8}  # H, C, N, O, F, P, S
        
        dist_matrix = torch.cdist(gen_coords, gen_coords)
        
        # Apply adaptive clash correction
        for i in range(len(gen_coords)):
            for j in range(i + 1, len(gen_coords)):
                atom_i_type = pred_atom_types[i].item()
                atom_j_type = pred_atom_types[j].item()
                
                # Get minimum allowed distance
                min_dist = vdw_radii.get(atom_i_type, 1.5) + vdw_radii.get(atom_j_type, 1.5) - 0.2
                current_dist = dist_matrix[i, j]
                
                if current_dist < min_dist:
                    # Move atoms apart
                    diff = gen_coords[i] - gen_coords[j]
                    if torch.norm(diff) > 1e-6:
                        direction = diff / torch.norm(diff)
                        correction = direction * (min_dist - current_dist) / 2
                        gen_coords[i] += correction
                        gen_coords[j] -= correction
        
        x_corrected[gen_mask] = gen_coords
        return x_corrected

    def enforce_bond_lengths_adaptive(self, x_pred, a_pred, edges, gen_mask):
        """Adaptive bond length enforcement based on atom types"""
        if edges.shape[1] == 0:
            return x_pred
            
        x_corrected = x_pred.clone()
        
        # Bond length lookup based on atom types
        bond_lengths = {
            (0, 1): 1.09, (1, 1): 1.54, (1, 2): 1.47, (1, 3): 1.43,  # H-C, C-C, C-N, C-O
            (2, 2): 1.45, (2, 3): 1.40, (3, 3): 1.48, (0, 2): 1.01, (0, 3): 0.96  # N-N, N-O, O-O, H-N, H-O
        }
        
        pred_atom_types = torch.argmax(a_pred, dim=-1)
        
        for edge_idx in range(edges.shape[1]):
            atom1_idx, atom2_idx = edges[0, edge_idx], edges[1, edge_idx]
            
            # Only adjust if at least one atom is generated
            if gen_mask[atom1_idx] or gen_mask[atom2_idx]:
                atom1_type = pred_atom_types[atom1_idx].item()
                atom2_type = pred_atom_types[atom2_idx].item()
                
                # Get target bond length
                bond_key = tuple(sorted([atom1_type, atom2_type]))
                target_length = bond_lengths.get(bond_key, 1.5)  # Default 1.5 Å
                
                diff = x_corrected[atom1_idx] - x_corrected[atom2_idx]
                current_length = torch.norm(diff)
                
                if current_length > 1e-6:
                    # Gentle adjustment toward target length
                    scale_factor = 0.8 * target_length / current_length + 0.2  # Weighted adjustment
                    center = (x_corrected[atom1_idx] + x_corrected[atom2_idx]) / 2
                    
                    x_corrected[atom1_idx] = center + diff * scale_factor / 2
                    x_corrected[atom2_idx] = center - diff * scale_factor / 2
        
        return x_corrected

    def fix_valence_violations_smart(self, x_pred, a_pred, gen_mask, edges):
        """Smart valence violation fixing with chemical knowledge"""
        if not gen_mask.any() or edges.shape[1] == 0:
            return a_pred
            
        a_corrected = a_pred.clone()
        
        # Map indices to types
        type_to_max_valence = {0: 1, 1: 4, 2: 3, 3: 2, 4: 1, 5: 5, 6: 6}  # H, C, N, O, F, P, S
        
        gen_indices = torch.where(gen_mask)[0]
        
        for atom_idx in gen_indices:
            # Count bonds for this atom
            edge_mask = (edges[0] == atom_idx) | (edges[1] == atom_idx)
            num_bonds = edge_mask.sum().item()
            
            # Get current predicted atom type
            current_probs = F.softmax(a_corrected[atom_idx], dim=-1)
            current_type = torch.argmax(current_probs).item()
            
            # Check if valence is violated
            max_valence = type_to_max_valence.get(current_type, 4)
            
            if num_bonds > max_valence:
                # Find suitable atom type that can accommodate this many bonds
                suitable_types = [t for t, v in type_to_max_valence.items() if v >= num_bonds]
                
                if suitable_types:
                    # Choose the most chemically reasonable type
                    # Preference: C > N > O > P > S > others
                    preference_order = [1, 2, 3, 5, 6, 0, 4]  # C, N, O, P, S, H, F
                    
                    best_type = None
                    for preferred in preference_order:
                        if preferred in suitable_types:
                            best_type = preferred
                            break
                    
                    if best_type is not None:
                        # Set new atom type with high confidence
                        new_probs = torch.zeros_like(current_probs)
                        new_probs[best_type] = 0.9
                        new_probs += 0.1 * current_probs  # Keep some of original distribution
                        new_probs = new_probs / new_probs.sum()
                        
                        a_corrected[atom_idx] = torch.log(new_probs + 1e-8)
        
        return a_corrected
    
    def format_output(self, X, S, A, generate_mask, block_ids, lengths):
        """Enhanced output formatting with better error handling"""
        batch_ids = length_to_batch_id(lengths)
        
        batch_S, batch_X, batch_A = [], [], []
        
        for i in range(len(lengths)):
            try:
                cur_batch_mask = (batch_ids == i)
                cur_gen_mask = generate_mask & cur_batch_mask
                
                if cur_gen_mask.any():
                    cur_s = S[cur_gen_mask].clamp(0, self.num_block_types - 1)
                    
                    # Get atoms belonging to generated blocks
                    gen_block_indices = torch.where(cur_gen_mask)[0]
                    cur_atom_mask = torch.zeros_like(block_ids, dtype=torch.bool)
                    
                    for block_idx in gen_block_indices:
                        cur_atom_mask |= (block_ids == block_idx)
                    
                    if cur_atom_mask.any():
                        cur_x = X[cur_atom_mask]
                        cur_a = A[cur_atom_mask].clamp(0, self.num_atom_types - 1)
                        
                        batch_S.append(cur_s.cpu().tolist())
                        batch_X.append([cur_x.cpu().tolist()])
                        batch_A.append([cur_a.cpu().tolist()])
                    else:
                        batch_S.append([])
                        batch_X.append([[]])
                        batch_A.append([[]])
                else:
                    batch_S.append([])
                    batch_X.append([[]])
                    batch_A.append([[]])
                    
            except Exception as e:
                print(f"Warning: Error formatting output for batch {i}: {e}")
                batch_S.append([])
                batch_X.append([[]])
                batch_A.append([[]])
        
        return batch_S, batch_X, batch_A