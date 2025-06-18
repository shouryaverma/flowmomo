import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
import numpy as np

from utils.gnn_utils import length_to_batch_id
from utils.chem_utils import (
    valence_check, cycle_check, check_stability, 
    connect_fragments, MAX_VALENCE, AtomVocab
)


class FlowMatching(nn.Module):
    def __init__(self, num_steps=50, sigma_min=1e-4, temp_schedule='cosine'):
        super().__init__()
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.temp_schedule = temp_schedule
        self.atom_vocab = AtomVocab()
        
        # DoPri5 coefficients (Dormand-Prince 5(4) method)
        self.dopri5_a = torch.tensor([
            [0, 0, 0, 0, 0, 0, 0],
            [1/5, 0, 0, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0, 0, 0],
            [44/45, -56/15, 32/9, 0, 0, 0, 0],
            [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
        ])
        
        self.dopri5_b = torch.tensor([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
        self.dopri5_b_hat = torch.tensor([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
        self.dopri5_c = torch.tensor([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
        
    def add_noise(self, X, S, A, generate_mask, block_ids, batch_ids, t=None):
        """Linear interpolation between noise and data (rectified flow)"""
        batch_size = batch_ids.max() + 1
        
        if t is None:
            t = torch.rand(batch_size, device=X.device)
        
        # Sample from prior (x_0) - more chemically informed
        x_0 = self.sample_chemical_prior(X, A, generate_mask, block_ids)
        s_0 = self.sample_discrete_prior(S, generate_mask)
        a_0 = self.sample_discrete_prior(A, generate_mask[block_ids])
        
        # Linear interpolation: x_t = (1-t) * x_0 + t * x_1
        # FIXED: Ensure proper indexing for batch IDs
        atom_batch_ids = batch_ids[block_ids]
        atom_batch_ids = torch.clamp(atom_batch_ids, 0, batch_size - 1)  # Safety clamp
        
        t_expand_atoms = t[atom_batch_ids]
        t_expand_blocks = t[batch_ids]
        
        x_t = (1 - t_expand_atoms[..., None]) * x_0 + t_expand_atoms[..., None] * X
        s_t = self.interpolate_discrete(s_0, S, t_expand_blocks)
        a_t = self.interpolate_discrete(a_0, A, t_expand_atoms)
        
        # Only interpolate in generation regions
        gen_mask_atoms = generate_mask[block_ids]
        x_t = torch.where(gen_mask_atoms[..., None], x_t, X)
        s_t = torch.where(generate_mask, s_t, S)
        a_t = torch.where(gen_mask_atoms, a_t, A)
        
        return t, x_t, s_t, a_t, x_0, s_0, a_0

    def sample_chemical_prior(self, X, A, generate_mask, block_ids):
        """Sample chemically-informed coordinate prior"""
        x_0 = torch.randn_like(X)
        
        # For generated atoms, add some structure based on nearby atoms
        gen_mask_atoms = generate_mask[block_ids]
        if gen_mask_atoms.any():
            # Use nearby context atoms to inform initial placement
            ctx_mask = ~gen_mask_atoms
            if ctx_mask.any():
                ctx_coords = X[ctx_mask]
                gen_coords = x_0[gen_mask_atoms]
                
                # Place generated atoms near context with some noise
                if len(ctx_coords) > 0:
                    # Simple approach: place near centroid of context
                    centroid = ctx_coords.mean(dim=0)
                    noise_scale = 2.0  # Angstroms
                    x_0[gen_mask_atoms] = centroid[None, :] + noise_scale * gen_coords
        
        return x_0

    def sample_discrete_prior(self, true_vals, mask):
        """Sample discrete variable prior with bounds checking"""
        if mask.any():
            # FIXED: Ensure reasonable bounds
            max_val = max(true_vals.max().item() + 1, 10)
            max_val = min(max_val, 1000)  # Prevent excessive vocabulary sizes
            
            prior_samples = torch.randint(0, max_val, true_vals.shape, device=true_vals.device)
            return torch.where(mask, prior_samples, true_vals)
        return true_vals
        
    def interpolate_discrete(self, x_0, x_1, t, temperature=None):
        """Improved discrete variable interpolation with better bounds checking"""
        if temperature is None:
            # Cosine temperature schedule - start hot, end cold
            if self.temp_schedule == 'cosine':
                temperature = 0.5 * (1 + torch.cos(torch.pi * t))
            else:
                temperature = torch.ones_like(t)
        
        # FIXED: Better bounds checking
        x_0_max = x_0.max().item() if len(x_0) > 0 else 0
        x_1_max = x_1.max().item() if len(x_1) > 0 else 0
        
        max_val = max(x_0_max, x_1_max) + 1
        max_val = max(max_val, 10)  # Ensure reasonable minimum
        max_val = min(max_val, 1000)  # Prevent excessive sizes
        
        # Clamp inputs to valid range
        x_0_clamped = torch.clamp(x_0, 0, max_val - 1)
        x_1_clamped = torch.clamp(x_1, 0, max_val - 1)
        
        x_0_oh = F.one_hot(x_0_clamped, max_val).float()
        x_1_oh = F.one_hot(x_1_clamped, max_val).float()
        
        # Linear interpolation in probability space
        x_t_logits = (1 - t[..., None]) * x_0_oh + t[..., None] * x_1_oh
        
        # Apply temperature for smoother interpolation
        x_t_logits = x_t_logits / (temperature[..., None] + 1e-8)
        
        # Sample from interpolated distribution
        x_t_probs = F.softmax(x_t_logits, dim=-1)
        x_t = torch.multinomial(x_t_probs + 1e-8, 1).squeeze(-1)
        
        return torch.clamp(x_t, 0, max_val - 1)

    def compute_velocity(self, model, t, x_t, s_t, a_t, bonds, position_ids, 
                        chain_ids, generate_mask, block_lengths, lengths, is_aa):
        """Compute velocity field for DoPri5 integration"""
        batch_ids = length_to_batch_id(lengths)
        block_ids = length_to_batch_id(block_lengths)
        
        # Forward pass through model
        H = model.embedding(s_t, a_t, block_ids)
        H = model.embed2hidden(H)
        
        # Add time conditioning
        atom_batch_ids = batch_ids[block_ids]
        max_batch_idx = len(t) - 1
        atom_batch_ids = torch.clamp(atom_batch_ids, 0, max_batch_idx)
        t_per_atom = t[atom_batch_ids]
        t_embed = model.time_embedding(t_per_atom)
        H = H + model.time_proj(t_embed)
        
        # Get edges for current state
        edges, edge_type = model.get_edges(
            batch_ids, chain_ids, x_t, block_ids, generate_mask, 
            position_ids, is_aa, block_lengths
        )
        edge_attr = model.edge_embedding(edge_type)
        
        # Encode then decode
        H_encoded, X_encoded = model.encoder(H, x_t, block_ids, batch_ids, edges, edge_attr)
        H_decoded, X_decoded = model.decoder(H_encoded, X_encoded, block_ids, batch_ids, edges, edge_attr)
        
        # Get velocity predictions
        v_x = model.coord_velocity_head(H_decoded)
        v_a = model.atom_velocity_head(H_decoded)
        v_s = model.block_velocity_head(H_decoded)
        v_s = scatter_mean(v_s, block_ids, dim=0)
        
        return v_x, v_a, v_s, edges

    def dopri5_step(self, model, t, dt, x_t, s_t, a_t, bonds, position_ids, 
                   chain_ids, generate_mask, block_lengths, lengths, is_aa, 
                   apply_constraints=True):
        """Single DoPri5 step with adaptive error control"""
        device = x_t.device
        batch_size = len(lengths)
        
        # Move coefficients to device
        a_coeffs = self.dopri5_a.to(device)
        b_coeffs = self.dopri5_b.to(device)
        b_hat_coeffs = self.dopri5_b_hat.to(device)
        c_coeffs = self.dopri5_c.to(device)
        
        # Store intermediate k values for coordinates
        k_x = torch.zeros(7, *x_t.shape, device=device)
        k_s = torch.zeros(7, *s_t.shape, device=device)
        k_a = torch.zeros(7, *a_t.shape, device=device)
        
        # Current time for batch
        t_current = torch.full((batch_size,), t, device=device)
        
        # Six stages of DoPri5
        for i in range(6):
            # Compute intermediate state
            x_temp = x_t.clone()
            s_temp = s_t.clone()
            a_temp = a_t.clone()
            
            for j in range(i):
                x_temp += dt * a_coeffs[i, j] * k_x[j]
                # For discrete variables, use smaller updates
                s_temp = self.update_discrete_rk(s_temp, k_s[j], dt * a_coeffs[i, j], generate_mask)
                a_temp = self.update_discrete_rk(a_temp, k_a[j], dt * a_coeffs[i, j], generate_mask[length_to_batch_id(block_lengths)])
            
            # Apply constraints to intermediate state
            if apply_constraints:
                x_temp, a_temp = self.apply_chemical_constraints(
                    x_temp, a_temp, s_temp, generate_mask, 
                    length_to_batch_id(block_lengths), length_to_batch_id(lengths),
                    chain_ids, torch.empty(2, 0, device=device)  # Dummy edges for now
                )
            
            # Compute velocity at intermediate point
            t_temp = t_current + c_coeffs[i] * dt
            v_x, v_a, v_s, edges = self.compute_velocity(
                model, t_temp, x_temp, s_temp, a_temp, bonds, position_ids,
                chain_ids, generate_mask, block_lengths, lengths, is_aa
            )
            
            # Store k values
            k_x[i] = v_x
            k_s[i] = v_s
            k_a[i] = v_a
        
        # Compute 5th order solution
        x_new = x_t.clone()
        s_new = s_t.clone()
        a_new = a_t.clone()
        
        for i in range(7):
            x_new += dt * b_coeffs[i] * k_x[i]
            s_new = self.update_discrete_rk(s_new, k_s[i], dt * b_coeffs[i], generate_mask)
            a_new = self.update_discrete_rk(a_new, k_a[i], dt * b_coeffs[i], generate_mask[length_to_batch_id(block_lengths)])
        
        # Compute 4th order solution for error estimation
        x_hat = x_t.clone()
        s_hat = s_t.clone()
        a_hat = a_t.clone()
        
        for i in range(7):
            x_hat += dt * b_hat_coeffs[i] * k_x[i]
            s_hat = self.update_discrete_rk(s_hat, k_s[i], dt * b_hat_coeffs[i], generate_mask)
            a_hat = self.update_discrete_rk(a_hat, k_a[i], dt * b_hat_coeffs[i], generate_mask[length_to_batch_id(block_lengths)])
        
        # Error estimation (only for coordinates - discrete variables are handled differently)
        gen_mask_atoms = generate_mask[length_to_batch_id(block_lengths)]
        if gen_mask_atoms.any():
            error_x = torch.norm(x_new - x_hat, dim=-1)[gen_mask_atoms]
            max_error = error_x.max().item() if len(error_x) > 0 else 0.0
        else:
            max_error = 0.0
        
        return x_new, s_new, a_new, max_error

    def update_discrete_rk(self, x_current, k_logits, dt_coeff, mask, temperature=0.1):
        """Update discrete variables in Runge-Kutta style"""
        if not mask.any() or torch.abs(dt_coeff) < 1e-12:
            return x_current
        
        # Convert logits to probabilities
        probs = F.softmax(k_logits / temperature, dim=-1)
        
        # Current state as one-hot
        if len(x_current.shape) == 1:  # If not one-hot yet
            x_current_oh = F.one_hot(x_current, probs.shape[-1]).float()
        else:
            x_current_oh = x_current
        
        # Small update toward velocity direction
        update_factor = torch.tanh(dt_coeff)  # Bound the update
        x_new_oh = x_current_oh + update_factor * (probs - x_current_oh)
        x_new_oh = x_new_oh / (x_new_oh.sum(-1, keepdim=True) + 1e-8)
        
        # Sample new discrete values
        x_new = torch.multinomial(x_new_oh + 1e-8, 1).squeeze(-1)
        
        # Only update in mask regions
        return torch.where(mask, x_new, x_current)

    @torch.no_grad()
    def sample_dopri5(self, model, X, S, A, bonds, position_ids, chain_ids, 
                     generate_mask, block_lengths, lengths, is_aa, 
                     rtol=1e-5, atol=1e-6, max_steps=1000, apply_constraints=True, **kwargs):
        """DoPri5 adaptive sampling for rectified flow with chemical constraints"""
        
        batch_ids = length_to_batch_id(lengths)
        block_ids = length_to_batch_id(block_lengths)
        
        # Initialize from chemically-informed noise in generation regions
        x_t = self.sample_chemical_prior(X, A, generate_mask, block_ids)
        x_t = torch.where(generate_mask[block_ids][..., None], x_t, X)
        
        s_t = self.sample_discrete_prior(S, generate_mask)
        a_t = self.sample_discrete_prior(A, generate_mask[block_ids])
        
        # Adaptive integration parameters
        t = 0.0
        dt = 0.01  # Initial step size
        dt_min = 1e-6
        dt_max = 0.1
        safety_factor = 0.9
        step_count = 0
        
        # Target tolerances
        atol = atol
        rtol = rtol
        
        while t < 1.0 and step_count < max_steps:
            # Ensure we don't overshoot
            if t + dt > 1.0:
                dt = 1.0 - t
            
            # Attempt a step
            x_new, s_new, a_new, error = self.dopri5_step(
                model, t, dt, x_t, s_t, a_t, bonds, position_ids,
                chain_ids, generate_mask, block_lengths, lengths, is_aa,
                apply_constraints
            )
            
            # Calculate tolerance
            scale = atol + rtol * torch.max(torch.abs(x_t), torch.abs(x_new))
            tolerance = scale.max().item()
            
            # Check if step is acceptable
            if error <= tolerance or dt <= dt_min:
                # Accept step
                x_t = x_new
                s_t = s_new
                a_t = a_new
                t += dt
                step_count += 1
                
                # Final constraint application
                if apply_constraints:
                    x_t, a_t = self.apply_chemical_constraints(
                        x_t, a_t, s_t, generate_mask, block_ids, batch_ids,
                        chain_ids, torch.empty(2, 0, device=x_t.device)
                    )
            
            # Adapt step size for next iteration
            if error > 1e-12:  # Avoid division by zero
                error_ratio = tolerance / error
                dt_new = dt * safety_factor * (error_ratio ** 0.2)  # 5th order method
                dt = torch.clamp(torch.tensor(dt_new), dt_min, dt_max).item()
            else:
                dt = min(dt * 2.0, dt_max)  # Increase step size if error is very small
        
        if step_count >= max_steps:
            print(f"Warning: DoPri5 reached maximum steps ({max_steps})")
        
        return self.format_output(x_t, s_t, a_t, generate_mask, block_ids, lengths)

    @torch.no_grad()
    def sample(self, model, X, S, A, bonds, position_ids, chain_ids, 
               generate_mask, block_lengths, lengths, is_aa, num_steps=50, 
               apply_constraints=True, use_dopri5=True, **kwargs):
        """Choose between Euler and DoPri5 sampling"""
        if use_dopri5:
            return self.sample_dopri5(
                model, X, S, A, bonds, position_ids, chain_ids,
                generate_mask, block_lengths, lengths, is_aa,
                apply_constraints=apply_constraints, **kwargs
            )
        else:
            return self.sample_euler(
                model, X, S, A, bonds, position_ids, chain_ids,
                generate_mask, block_lengths, lengths, is_aa,
                num_steps, apply_constraints, **kwargs
            )
    
    @torch.no_grad()
    def sample_euler(self, model, X, S, A, bonds, position_ids, chain_ids, 
                    generate_mask, block_lengths, lengths, is_aa, num_steps=50, 
                    apply_constraints=True, **kwargs):
        """Original Euler sampling method (kept for comparison)"""
        
        batch_ids = length_to_batch_id(lengths)
        block_ids = length_to_batch_id(block_lengths)
        
        # Initialize from chemically-informed noise in generation regions
        x_t = self.sample_chemical_prior(X, A, generate_mask, block_ids)
        x_t = torch.where(generate_mask[block_ids][..., None], x_t, X)
        
        s_t = self.sample_discrete_prior(S, generate_mask)
        a_t = self.sample_discrete_prior(A, generate_mask[block_ids])
        
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((batch_ids.max() + 1,), i * dt, device=X.device)
            
            # Get current temperature for discrete updates
            temp = self.get_temperature(i, num_steps)
            
            # Compute velocities
            v_x, v_a, v_s, edges = self.compute_velocity(
                model, t, x_t, s_t, a_t, bonds, position_ids,
                chain_ids, generate_mask, block_lengths, lengths, is_aa
            )
            
            # Update using Euler method
            x_t_new = x_t + v_x * dt
            s_t_new = self.update_discrete(s_t, v_s, dt, generate_mask, temp)
            a_t_new = self.update_discrete(a_t, v_a, dt, generate_mask[block_ids], temp)
            
            # Apply chemical constraints if enabled
            if apply_constraints:
                x_t_new, a_t_new = self.apply_chemical_constraints(
                    x_t_new, a_t_new, s_t_new, generate_mask, block_ids, 
                    batch_ids, chain_ids, edges
                )
            
            # Update states
            x_t = x_t_new
            s_t = s_t_new  
            a_t = a_t_new
            
        return self.format_output(x_t, s_t, a_t, generate_mask, block_ids, lengths)

    def get_temperature(self, step, total_steps):
        """Temperature schedule for discrete variable updates"""
        progress = step / total_steps
        
        if self.temp_schedule == 'cosine':
            return 0.1 + 0.9 * (1 + np.cos(np.pi * progress)) / 2
        elif self.temp_schedule == 'linear':
            return 1.0 - 0.9 * progress
        else:
            return 1.0
    
    def update_discrete(self, x_t, v_logits, dt, mask, temperature=1.0):
        """Update discrete variables with temperature control"""
        if not mask.any():
            return x_t
            
        # Apply temperature to logits
        v_logits_temp = v_logits / (temperature + 1e-8)
        probs = F.softmax(v_logits_temp, dim=-1)
        
        # Current state as one-hot
        x_t_oh = F.one_hot(x_t, probs.shape[-1]).float()
        
        # Interpolate toward velocity direction
        x_t_oh_new = x_t_oh + dt * (probs - x_t_oh)
        x_t_oh_new = x_t_oh_new / (x_t_oh_new.sum(-1, keepdim=True) + 1e-8)
        
        # Sample new discrete values
        x_new = torch.multinomial(x_t_oh_new + 1e-8, 1).squeeze(-1)
        
        # Only update in mask regions
        return torch.where(mask, x_new, x_t)

    def apply_chemical_constraints(self, x_pred, a_pred, s_pred, generate_mask, 
                                 block_ids, batch_ids, chain_ids, edges):
        """Apply chemical constraints during sampling"""
        gen_mask_atoms = generate_mask[block_ids]
        if not gen_mask_atoms.any():
            return x_pred, a_pred
        
        x_corrected = x_pred.clone()
        a_corrected = a_pred.clone()
        
        try:
            # 1. Avoid atomic clashes
            x_corrected = self.avoid_clashes(x_corrected, a_corrected, gen_mask_atoms)
            
            # 2. Enforce reasonable bond lengths
            if edges.shape[1] > 0:
                x_corrected = self.enforce_bond_lengths(x_corrected, a_corrected, edges, gen_mask_atoms)
            
            # 3. Check and fix valence violations
            a_corrected = self.fix_valence_violations(x_corrected, a_corrected, gen_mask_atoms, edges)
            
        except Exception as e:
            # If constraint application fails, return original predictions
            print(f"Warning: Constraint application failed: {e}")
            return x_pred, a_pred
        
        return x_corrected, a_corrected

    def avoid_clashes(self, x_pred, a_pred, gen_mask, min_distance=1.0):
        """Push apart atoms that are too close"""
        if not gen_mask.any():
            return x_pred
            
        x_corrected = x_pred.clone()
        gen_coords = x_corrected[gen_mask]
        
        if len(gen_coords) > 1:
            dist_matrix = torch.cdist(gen_coords, gen_coords)
            
            # Find pairs that are too close
            close_pairs = (dist_matrix < min_distance) & (dist_matrix > 0)
            
            if close_pairs.any():
                # Simple repulsion: move atoms apart along their connection vector
                for i in range(len(gen_coords)):
                    for j in range(i + 1, len(gen_coords)):
                        if close_pairs[i, j]:
                            diff = gen_coords[i] - gen_coords[j]
                            dist = torch.norm(diff)
                            if dist > 1e-6:
                                direction = diff / dist
                                correction = direction * (min_distance - dist) / 2
                                gen_coords[i] += correction
                                gen_coords[j] -= correction
                
                x_corrected[gen_mask] = gen_coords
        
        return x_corrected

    def enforce_bond_lengths(self, x_pred, a_pred, edges, gen_mask, target_length=1.5):
        """Adjust coordinates to maintain reasonable bond lengths"""
        if edges.shape[1] == 0:
            return x_pred
            
        x_corrected = x_pred.clone()
        
        for edge_idx in range(edges.shape[1]):
            atom1_idx, atom2_idx = edges[0, edge_idx], edges[1, edge_idx]
            
            # Only adjust if at least one atom is generated
            if gen_mask[atom1_idx] or gen_mask[atom2_idx]:
                diff = x_corrected[atom1_idx] - x_corrected[atom2_idx]
                current_length = torch.norm(diff)
                
                if current_length > 1e-6:
                    # Adjust toward target bond length
                    scale_factor = target_length / current_length
                    center = (x_corrected[atom1_idx] + x_corrected[atom2_idx]) / 2
                    
                    # Move atoms to achieve target distance
                    direction = diff / current_length
                    x_corrected[atom1_idx] = center + direction * target_length / 2
                    x_corrected[atom2_idx] = center - direction * target_length / 2
        
        return x_corrected

    def fix_valence_violations(self, x_pred, a_pred, gen_mask, edges):
        """Adjust atom types to fix valence violations"""
        if not gen_mask.any() or edges.shape[1] == 0:
            return a_pred
            
        a_corrected = a_pred.clone()
        
        # Count bonds for each generated atom
        gen_indices = torch.where(gen_mask)[0]
        
        for atom_idx in gen_indices:
            # Count edges involving this atom
            edge_mask = (edges[0] == atom_idx) | (edges[1] == atom_idx)
            num_bonds = edge_mask.sum().item()
            
            # Get current predicted atom type
            current_type_idx = torch.argmax(a_corrected[atom_idx]).item()
            
            if current_type_idx < len(self.atom_vocab.idx2atom):
                current_symbol = self.atom_vocab.idx2atom[current_type_idx]
                
                # If valence is violated, try to find a compatible atom type
                if current_symbol in MAX_VALENCE and num_bonds > MAX_VALENCE[current_symbol]:
                    # Find an atom type that can accommodate this many bonds
                    for symbol, max_val in MAX_VALENCE.items():
                        if max_val >= num_bonds:
                            try:
                                new_type_idx = self.atom_vocab.atom2idx[symbol]
                                # Set this as the most likely atom type
                                a_corrected[atom_idx] = 0
                                a_corrected[atom_idx, new_type_idx] = 1.0
                                break
                            except KeyError:
                                continue
        
        return a_corrected
    
    def format_output(self, X, S, A, generate_mask, block_ids, lengths):
        """Format output similar to existing codebase"""
        batch_ids = length_to_batch_id(lengths)
        
        batch_S, batch_X, batch_A = [], [], []
        
        for i, l in enumerate(lengths):
            cur_batch_mask = (batch_ids == i)
            cur_gen_mask = generate_mask & cur_batch_mask
            
            if cur_gen_mask.any():
                cur_s = S[cur_gen_mask]
                
                # Get atoms belonging to generated blocks
                gen_block_indices = torch.where(cur_gen_mask)[0]
                cur_atom_mask = torch.zeros_like(block_ids, dtype=torch.bool)
                
                for block_idx in gen_block_indices:
                    cur_atom_mask |= (block_ids == block_idx)
                
                if cur_atom_mask.any():
                    cur_x = X[cur_atom_mask]
                    cur_a = A[cur_atom_mask]
                    
                    batch_S.append(cur_s.tolist())
                    batch_X.append([cur_x.tolist()])
                    batch_A.append([cur_a.tolist()])
                else:
                    batch_S.append([])
                    batch_X.append([[]])
                    batch_A.append([[]])
            else:
                batch_S.append([])
                batch_X.append([[]])
                batch_A.append([[]])
        
        return batch_S, batch_X, batch_A