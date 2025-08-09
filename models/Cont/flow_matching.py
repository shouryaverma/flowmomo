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
    Similar to EpsilonNet but predicts velocities instead of noise.
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
        Returns:
            v_H: (N, hidden_size) - predicted velocity for H
            v_X: (N, 3) - predicted velocity for X
        """
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
    Simplified Flow Matching for molecular design.
    FIXED: Removed contrastive learning, uses simple MSE loss like LDM.
    """
    
    def __init__(
        self, 
        latent_size,
        hidden_size,
        encoder_type='EPT',
        encoder_opt={},
        sigma=1e-4,
        lambda_contrast=0.0,  # FIXED: Default to 0 (no contrastive)
    ):
        super().__init__()
        self.velocity_net = VelocityNet(latent_size, hidden_size, encoder_type, encoder_opt)
        self.sigma = float(sigma)
        self.lambda_contrast = float(lambda_contrast)

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

    def get_path_and_velocity(self, H_0, X_0, H_1, X_1, t, generate_mask):
        """
        FIXED: Simple path and velocity computation
        """
        # Ensure proper time dimension handling
        if t.dim() == 1:
            t_H = t.unsqueeze(-1)  # [N, 1] for H features
            t_X = t.unsqueeze(-1)  # [N, 1] for X features  
        else:
            t_H = t
            t_X = t
        
        # Linear interpolation: x_t = (1-t) * x_0 + t * x_1
        H_t = (1 - t_H) * H_0 + t_H * H_1
        X_t = (1 - t_X) * X_0 + t_X * X_1
        
        # Target velocity for rectified flow: v = x_1 - x_0
        v_H_target = H_1 - H_0
        v_X_target = X_1 - X_0

        # Clamp extreme velocities
        v_H_target = torch.clamp(v_H_target, min=-10.0, max=10.0)
        v_X_target = torch.clamp(v_X_target, min=-10.0, max=10.0)

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

    def contrastive_flow_matching_loss(self, H_0, X_0, H_1, X_1, cond_embedding, chain_ids, generate_mask, lengths):
        """
        FIXED: Simplified flow matching loss using MSE like LDM (no contrastive)
        """
        batch_ids = length_to_batch_id(lengths)
        batch_size = batch_ids.max() + 1
        
        # Check for generation targets
        if generate_mask.sum() == 0:
            zero_tensor = torch.tensor(0.0, device=H_0.device, requires_grad=True)
            return {
                'H': zero_tensor, 'X': zero_tensor,
                'H_pos': zero_tensor, 'X_pos': zero_tensor,
                'H_neg': zero_tensor, 'X_neg': zero_tensor,
            }
        
        # Sample time with numerical stability
        t = torch.rand(batch_size, device=H_0.device)
        t = torch.clamp(t, min=1e-3, max=1.0-1e-3)  # Avoid exact 0 and 1
        t = t[batch_ids]  # Expand to node level
        
        # Get path and velocity
        H_t, X_t, v_H_target, v_X_target = self.get_path_and_velocity(
            H_0, X_0, H_1, X_1, t, generate_mask
        )
        
        # Add conditional noise for stability
        if self.sigma > 0:
            H_t = H_t + self.sigma * torch.randn_like(H_t)
            X_t = X_t + self.sigma * torch.randn_like(X_t)
        
        # Get edges and predict velocity
        edges, edge_types = self._get_edges(chain_ids, batch_ids, lengths)
        v_H_pred, v_X_pred = self.velocity_net(
            H_t, X_t, cond_embedding, edges, edge_types, generate_mask, batch_ids, t
        )
        
        # FIXED: Simple MSE loss like LDM
        gen_mask = generate_mask
        if gen_mask.sum() > 0:
            # Use simple MSE reduction='none' then mean like LDM
            h_error = F.mse_loss(v_H_pred[gen_mask], v_H_target[gen_mask], reduction='none')
            x_error = F.mse_loss(v_X_pred[gen_mask], v_X_target[gen_mask], reduction='none')
            
            # Sum over feature dimensions and average over nodes (like LDM)
            loss_H = h_error.sum(-1).mean()
            loss_X = x_error.sum(-1).mean()
        else:
            loss_H = torch.tensor(0.0, device=H_0.device, requires_grad=True)
            loss_X = torch.tensor(0.0, device=H_0.device, requires_grad=True)
        
        # FIXED: No contrastive learning (simplified)
        return {
            'H': loss_H,
            'X': loss_X,
            'H_pos': loss_H,  # For compatibility
            'X_pos': loss_X,  # For compatibility
            'H_neg': torch.tensor(0.0, device=H_0.device),  # Zero for no contrastive
            'X_neg': torch.tensor(0.0, device=H_0.device),  # Zero for no contrastive
        }

    @torch.no_grad()
    def sample_ode(self, H_init, X_init, cond_embedding, chain_ids, generate_mask, 
                   lengths, num_steps=50, method='euler', pbar=False):
        """
        FIXED: Improved ODE sampling with better numerical stability
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
        
        for step in pbar_fn(range(num_steps)):
            # Use midpoint time for better accuracy
            t = torch.full_like(batch_ids, (step + 0.5) / num_steps, dtype=torch.float)
            
            if method == 'euler':
                # Euler integration
                v_H, v_X = self.velocity_net(
                    H_t, X_t, cond_embedding, edges, edge_types, generate_mask, batch_ids, t
                )
                H_t = H_t + dt * v_H
                X_t = X_t + dt * v_X
                
            elif method == 'rk4':
                # Runge-Kutta 4th order
                def get_velocity(H, X, time):
                    return self.velocity_net(
                        H, X, cond_embedding, edges, edge_types, generate_mask, batch_ids, time
                    )
                
                # RK4 integration
                k1_H, k1_X = get_velocity(H_t, X_t, t)
                k2_H, k2_X = get_velocity(H_t + 0.5*dt*k1_H, X_t + 0.5*dt*k1_X, t + 0.5*dt)
                k3_H, k3_X = get_velocity(H_t + 0.5*dt*k2_H, X_t + 0.5*dt*k2_X, t + 0.5*dt)
                k4_H, k4_X = get_velocity(H_t + dt*k3_H, X_t + dt*k3_X, t + dt)
                
                H_t = H_t + (dt/6) * (k1_H + 2*k2_H + 2*k3_H + k4_H)
                X_t = X_t + (dt/6) * (k1_X + 2*k2_X + 2*k3_X + k4_X)
                
            else:
                raise ValueError(f"Unknown integration method: {method}")
            
            # Apply generation mask
            H_t = torch.where(generate_mask[:, None].expand_as(H_t), H_t, H_init)
            X_t = torch.where(generate_mask[:, None].expand_as(X_t), X_t, X_init)
            
            # Add numerical stability clipping
            H_t = torch.clamp(H_t, -10, 10)
            X_t = torch.clamp(X_t, -10, 10)
            
            traj[step + 1] = (H_t.clone(), X_t.clone())
        
        return traj

    def forward(self, H_0, X_0, H_1, X_1, cond_embedding, chain_ids, generate_mask, lengths):
        """
        Forward pass for training.
        """
        return self.contrastive_flow_matching_loss(
            H_0, X_0, H_1, X_1, cond_embedding, chain_ids, generate_mask, lengths
        )