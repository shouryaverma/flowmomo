#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from tqdm.auto import tqdm
from torch_scatter import scatter_mean
from utils.gnn_utils import variadic_meshgrid, length_to_batch_id
from utils.nn_utils import SinusoidalTimeEmbeddings
from ..modules.create_net import create_net
from ..modules.nn import MLP


class VelocityNet(nn.Module):
    """
    Neural network that predicts the velocity field for flow matching.
    Enhanced with classifier-free guidance support.
    """
    
    def __init__(
            self,
            input_size,
            hidden_size,
            encoder_type='EPT',
            opt={'n_layers': 3}
        ):
        super().__init__()
        
        edge_embed_size = hidden_size // 4
        self.input_mlp = MLP(
            input_size + hidden_size * 2,  # latent variable, cond embedding, time embedding
            hidden_size, hidden_size, 3
        )
        self.encoder = create_net(encoder_type, hidden_size, edge_embed_size, opt)
        self.hidden2input = nn.Linear(hidden_size, input_size)
        self.edge_embedding = nn.Embedding(2, edge_embed_size)
        self.time_embedding = SinusoidalTimeEmbeddings(hidden_size)

    def forward(
            self,
            H_t,
            X_t,
            cond_embedding,
            edges,
            edge_types,
            generate_mask,
            batch_ids,
            t,
            guidance_scale=1.0,
            use_guidance=False
        ):
        """
        Args:
            H_t: (N, hidden_size) - latent features at time t
            X_t: (N, 3) - coordinates at time t
            cond_embedding: (N, hidden_size) - conditional embedding
            edges: (2, E) - edge indices
            edge_types: (E,) - edge types
            generate_mask: (N,) - mask for generation
            batch_ids: (N,) - batch indices
            t: (N,) - time values [0, 1]
            guidance_scale: float - strength of classifier-free guidance
            use_guidance: bool - whether to apply classifier-free guidance
        Returns:
            v_H: (N, hidden_size) - predicted velocity for H
            v_X: (N, 3) - predicted velocity for X
        """
        
        if use_guidance and guidance_scale != 1.0:
            # Conditional prediction
            v_H_cond, v_X_cond = self._predict_velocity(
                H_t, X_t, cond_embedding, edges, edge_types, generate_mask, batch_ids, t
            )
            
            # Unconditional prediction (zero out conditioning)
            zero_cond = torch.zeros_like(cond_embedding)
            v_H_uncond, v_X_uncond = self._predict_velocity(
                H_t, X_t, zero_cond, edges, edge_types, generate_mask, batch_ids, t
            )
            
            # Apply classifier-free guidance
            v_H = v_H_uncond + guidance_scale * (v_H_cond - v_H_uncond)
            v_X = v_X_uncond + guidance_scale * (v_X_cond - v_X_uncond)
            
            return v_H, v_X
        else:
            # Standard prediction without guidance
            return self._predict_velocity(
                H_t, X_t, cond_embedding, edges, edge_types, generate_mask, batch_ids, t
            )
    
    def _predict_velocity(self, H_t, X_t, cond_embedding, edges, edge_types, generate_mask, batch_ids, t):
        """Core velocity prediction logic"""
        t_embed = self.time_embedding(t)
        in_feat = torch.cat([H_t, cond_embedding, t_embed], dim=-1)
        in_feat = self.input_mlp(in_feat)
        edge_embed = self.edge_embedding(edge_types)
        block_ids = torch.arange(in_feat.shape[0], device=in_feat.device)
        
        out_H, out_X = self.encoder(in_feat, X_t, block_ids, batch_ids, edges, edge_embed)

        # Velocity for coordinates (equivariant)
        v_X = out_X
        v_X = torch.where(generate_mask[:, None].expand_as(v_X), v_X, torch.zeros_like(v_X))

        # Velocity for latent features (invariant)
        out_H = self.hidden2input(out_H)
        v_H = out_H
        v_H = torch.where(generate_mask[:, None].expand_as(v_H), v_H, torch.zeros_like(v_H))

        return v_H, v_X


class FlowMatching(nn.Module):
    """
    Fixed Flow Matching for molecular latent spaces.
    """
    
    def __init__(
        self, 
        latent_size,
        hidden_size,
        encoder_type='EPT',
        encoder_opt={},
        sigma=1e-4,
        lambda_contrast=0.0,
        p_uncond=0.1,
        use_ot_coupling=True,
        time_weighting='uniform',  # Start with uniform, not u_shaped
    ):
        super().__init__()
        self.velocity_net = VelocityNet(latent_size, hidden_size, encoder_type, encoder_opt)
        self.sigma = float(sigma)
        self.lambda_contrast = float(lambda_contrast)
        self.p_uncond = float(p_uncond)
        self.use_ot_coupling = use_ot_coupling
        self.time_weighting = time_weighting

    def sample_molecular_prior(self, H_shape, X_shape, device, block_lengths=None):
        """
        Sample priors scaled by molecular size - FIXED VERSION
        """
        if block_lengths is not None:
            # Scale noise by molecule size (number of blocks)
            H_scale = torch.sqrt(torch.tensor(H_shape[0], device=device, dtype=torch.float)) * 0.1
            X_scale = torch.sqrt(torch.tensor(X_shape[0], device=device, dtype=torch.float)) * 0.1
        else:
            H_scale, X_scale = 1.0, 1.0
        
        # Sample from scaled Gaussian
        H_0 = torch.randn(H_shape, device=device) * H_scale
        X_0 = torch.randn(X_shape, device=device) * X_scale
        
        return H_0, X_0

    def assignment_coupling(self, H_0, X_0, H_1, X_1):
        """
        Optimal transport using assignment - FIXED to always work
        """
        batch_size = H_0.shape[0]
        
        if batch_size <= 1:
            # No coupling needed for single sample
            return H_0, X_0, H_1, X_1
        
        # Compute cost matrix
        H_cost = torch.cdist(H_0.view(batch_size, -1), H_1.view(batch_size, -1), p=2) ** 2
        X_cost = torch.cdist(X_0.view(batch_size, -1), X_1.view(batch_size, -1), p=2) ** 2
        
        # Combined cost (normalize by dimensions)
        cost = H_cost / H_0.shape[-1] + X_cost / X_0.shape[-1]
        
        # Greedy assignment (find minimum cost pairing)
        _, indices = cost.min(dim=1)
        
        return H_0, X_0, H_1[indices], X_1[indices]

    def get_path_and_velocity(self, H_0, X_0, H_1, X_1, t, generate_mask):
        """
        Interpolation with gentle noise - FIXED VERSION
        """
        # Ensure proper time dimension handling
        if t.dim() == 1:
            t_H = t.unsqueeze(-1)  # [N, 1] for H features
            t_X = t.unsqueeze(-1)  # [N, 1] for X features  
        else:
            t_H = t
            t_X = t
        
        # Linear interpolation with gentle noise
        H_t = (1 - t_H) * H_0 + t_H * H_1
        X_t = (1 - t_X) * X_0 + t_X * X_1
        
        # Add gentle Gaussian noise for stability
        noise_scale = 0.01 * torch.sqrt(t_H * (1 - t_H))  # Maximum at t=0.5
        H_t = H_t + noise_scale * torch.randn_like(H_t)
        
        noise_scale_x = 0.01 * torch.sqrt(t_X * (1 - t_X))
        X_t = X_t + noise_scale_x * torch.randn_like(X_t)
        
        # Target velocity for rectified flow: v = x_1 - x_0
        v_H_target = H_1 - H_0
        v_X_target = X_1 - X_0
        
        # Apply generation mask (only modify generated parts)
        if generate_mask is not None:
            gen_mask_H = generate_mask.unsqueeze(-1).expand_as(H_t)
            gen_mask_X = generate_mask.unsqueeze(-1).expand_as(X_t)
            
            # For context parts, use target state directly and zero velocity
            H_t = torch.where(gen_mask_H, H_t, H_1)
            X_t = torch.where(gen_mask_X, X_t, X_1)
            v_H_target = torch.where(gen_mask_H, v_H_target, torch.zeros_like(v_H_target))
            v_X_target = torch.where(gen_mask_X, v_X_target, torch.zeros_like(v_X_target))
        
        return H_t, X_t, v_H_target, v_X_target

    @torch.no_grad()
    def _get_edges(self, chain_ids, batch_ids, lengths):
        """Generate edges for the molecular graph."""
        row, col = variadic_meshgrid(
            input1=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size1=lengths,
            input2=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size2=lengths,
        )
        
        is_ctx = chain_ids[row] == chain_ids[col]
        is_inter = ~is_ctx
        ctx_edges = torch.stack([row[is_ctx], col[is_ctx]], dim=0)
        inter_edges = torch.stack([row[is_inter], col[is_inter]], dim=0)
        edges = torch.cat([ctx_edges, inter_edges], dim=-1)
        edge_types = torch.cat([torch.zeros_like(ctx_edges[0]), torch.ones_like(inter_edges[0])], dim=0)
        return edges, edge_types

    def contrastive_flow_matching_loss(self, H_0, X_0, H_1, X_1, cond_embedding, chain_ids, generate_mask, lengths):
        """
        FIXED flow matching loss for molecular latents
        """
        batch_ids = length_to_batch_id(lengths)
        batch_size = batch_ids.max() + 1
        
        # Check for generation targets
        if generate_mask.sum() == 0:
            zero_tensor = torch.tensor(0.0, device=H_1.device, requires_grad=True)
            return {
                'H': zero_tensor, 'X': zero_tensor,
                'H_pos': zero_tensor, 'X_pos': zero_tensor,
                'H_neg': zero_tensor, 'X_neg': zero_tensor,
            }
        
        # FIXED: Simple uniform time sampling
        t = torch.rand(batch_size, device=H_1.device)
        t = torch.clamp(t, min=1e-3, max=1.0-1e-3)  # Numerical stability
        t_node = t[batch_ids]  # Expand to node level
        
        # FIXED: Use molecular-aware prior sampling
        H_0, X_0 = self.sample_molecular_prior(H_1.shape, X_1.shape, H_1.device)
        
        # FIXED: Always apply OT coupling when enabled
        if self.use_ot_coupling and generate_mask.sum() > 1:
            # Apply OT coupling only to generated parts
            gen_indices = torch.nonzero(generate_mask, as_tuple=False).squeeze(-1)
            
            if len(gen_indices) > 1:
                H_0_gen, X_0_gen, H_1_gen, X_1_gen = self.assignment_coupling(
                    H_0[gen_indices], X_0[gen_indices], 
                    H_1[gen_indices], X_1[gen_indices]
                )
                
                # Update only the generated parts
                H_0[gen_indices] = H_0_gen
                X_0[gen_indices] = X_0_gen
                H_1[gen_indices] = H_1_gen  
                X_1[gen_indices] = X_1_gen
        
        # Get path and velocity with gentle noise
        H_t, X_t, v_H_target, v_X_target = self.get_path_and_velocity(
            H_0, X_0, H_1, X_1, t_node, generate_mask
        )
        
        # Classifier-free guidance training
        if self.training and self.p_uncond > 0:
            batch_mask = torch.rand(batch_size, device=H_1.device) < self.p_uncond
            node_mask = batch_mask[batch_ids].unsqueeze(-1)
            cond_embedding = torch.where(
                node_mask.expand_as(cond_embedding),
                torch.zeros_like(cond_embedding),
                cond_embedding
            )
        
        # Get edges and predict velocity
        edges, edge_types = self._get_edges(chain_ids, batch_ids, lengths)
        v_H_pred, v_X_pred = self.velocity_net(
            H_t, X_t, cond_embedding, edges, edge_types, generate_mask, batch_ids, t_node,
            guidance_scale=1.0, use_guidance=False
        )
        
        # FIXED: Simple MSE loss without complex weighting
        gen_mask = generate_mask
        if gen_mask.sum() > 0:
            # Simple MSE loss
            loss_H = F.mse_loss(v_H_pred[gen_mask], v_H_target[gen_mask])
            loss_X = F.mse_loss(v_X_pred[gen_mask], v_X_target[gen_mask])
        else:
            loss_H = torch.tensor(0.0, device=H_1.device, requires_grad=True)
            loss_X = torch.tensor(0.0, device=H_1.device, requires_grad=True)
        
        return {
            'H': loss_H,
            'X': loss_X,
            'H_pos': loss_H,
            'X_pos': loss_X,
            'H_neg': torch.tensor(0.0, device=H_1.device),
            'X_neg': torch.tensor(0.0, device=H_1.device),
        }

    @torch.no_grad()
    def sample_ode(self, H_init, X_init, cond_embedding, chain_ids, generate_mask, 
                   lengths, num_steps=50, method='euler', pbar=False, guidance_scale=1.0):
        """
        FIXED ODE sampling for molecular generation
        """
        batch_ids = length_to_batch_id(lengths)
        dt = 1.0 / num_steps
        
        # Get edges
        edges, edge_types = self._get_edges(chain_ids, batch_ids, lengths)
        
        # Initialize trajectory
        traj = {0: (H_init.clone(), X_init.clone())}
        
        if pbar:
            pbar_fn = functools.partial(tqdm, total=num_steps, desc='Flow Sampling')
        else:
            pbar_fn = lambda x: x
        
        H_t, X_t = H_init.clone(), X_init.clone()
        
        # Determine if we should use guidance
        use_guidance = guidance_scale != 1.0
        
        for step in pbar_fn(range(num_steps)):
            # Use midpoint time for better accuracy
            t = torch.full_like(batch_ids, (step + 0.5) / num_steps, dtype=torch.float)
            
            # Euler integration with optional classifier-free guidance
            v_H, v_X = self.velocity_net(
                H_t, X_t, cond_embedding, edges, edge_types, generate_mask, batch_ids, t,
                guidance_scale=guidance_scale, use_guidance=use_guidance
            )
            H_t = H_t + dt * v_H
            X_t = X_t + dt * v_X
            
            # Apply generation mask
            H_t = torch.where(generate_mask[:, None].expand_as(H_t), H_t, H_init)
            X_t = torch.where(generate_mask[:, None].expand_as(X_t), X_t, X_init)
            
            # Add numerical stability clipping
            H_t = torch.clamp(H_t, -5, 5)  # Less aggressive clipping
            X_t = torch.clamp(X_t, -5, 5)
            
            traj[step + 1] = (H_t.clone(), X_t.clone())
        
        return traj

    # REMOVED: All the complex statistics tracking and time weighting
    # REMOVED: sample_matched_prior with global statistics
    # REMOVED: sinkhorn_coupling (too complex)
    # REMOVED: complex time weighting functions

    def forward(self, H_0, X_0, H_1, X_1, cond_embedding, chain_ids, generate_mask, lengths):
        """
        Forward pass for training.
        """
        return self.contrastive_flow_matching_loss(
            H_0, X_0, H_1, X_1, cond_embedding, chain_ids, generate_mask, lengths
        )