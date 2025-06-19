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
    Core Flow Matching framework that handles the coupling between source and target distributions.
    """
    
    def __init__(
        self, 
        latent_size,
        hidden_size,
        encoder_type='EPT',
        encoder_opt={},
        sigma=1e-4,  # noise level for conditional flow matching
    ):
        super().__init__()
        self.velocity_net = VelocityNet(latent_size, hidden_size, encoder_type, encoder_opt)
        self.sigma = float(sigma)

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

    def sample_time(self, batch_size, device):
        """Sample random time points."""
        return torch.rand(batch_size, device=device)

    def get_path_and_velocity(self, H_0, X_0, H_1, X_1, t, generate_mask):
        """
        Compute the path and target velocity for flow matching.
        
        Args:
            H_0, X_0: source (noise)
            H_1, X_1: target (data) 
            t: time [0, 1]
            generate_mask: mask for generation
            
        Returns:
            H_t, X_t: interpolated states
            v_H_target, v_X_target: target velocities
        """
        # Expand time to match feature dimensions
        t_H = t.view(-1, 1).expand_as(H_0)
        t_X = t.view(-1, 1).expand_as(X_0)
        
        # Linear interpolation path: x_t = (1-t) * x_0 + t * x_1
        H_t = (1 - t_H) * H_0 + t_H * H_1
        X_t = (1 - t_X) * X_0 + t_X * X_1
        
        # Target velocity: v = x_1 - x_0 (for rectified flow)
        v_H_target = H_1 - H_0
        v_X_target = X_1 - X_0
        
        # Apply generation mask
        H_t = torch.where(generate_mask[:, None].expand_as(H_t), H_t, H_1)
        X_t = torch.where(generate_mask[:, None].expand_as(X_t), X_t, X_1)
        
        return H_t, X_t, v_H_target, v_X_target

    def conditional_flow_matching_loss(self, H_0, X_0, H_1, X_1, cond_embedding, 
                                     chain_ids, generate_mask, lengths):
        """
        Compute the conditional flow matching loss.
        
        Args:
            H_0, X_0: source distribution samples (noise)
            H_1, X_1: target distribution samples (data)
            cond_embedding: conditional information
            chain_ids: chain identifiers
            generate_mask: generation mask
            lengths: batch lengths
            
        Returns:
            loss_dict: dictionary containing losses
        """
        batch_ids = length_to_batch_id(lengths)
        batch_size = batch_ids.max() + 1
        
        # Sample time uniformly
        t = self.sample_time(batch_size, H_0.device)[batch_ids]
        
        # Get interpolated path and target velocity
        H_t, X_t, v_H_target, v_X_target = self.get_path_and_velocity(
            H_0, X_0, H_1, X_1, t, generate_mask
        )
        
        # Add conditional noise for stability (conditional flow matching)
        if self.sigma > 0:
            H_t = H_t + self.sigma * torch.randn_like(H_t)
            X_t = X_t + self.sigma * torch.randn_like(X_t)
        
        # Get edges
        edges, edge_types = self._get_edges(chain_ids, batch_ids, lengths)
        
        # Predict velocity
        v_H_pred, v_X_pred = self.velocity_net(
            H_t, X_t, cond_embedding, edges, edge_types, generate_mask, batch_ids, t
        )
        
        # Compute losses
        loss_H = F.mse_loss(v_H_pred[generate_mask], v_H_target[generate_mask], reduction='none').sum(dim=-1)
        loss_H = loss_H.sum() / (generate_mask.sum().float() + 1e-8)
        
        loss_X = F.mse_loss(v_X_pred[generate_mask], v_X_target[generate_mask], reduction='none').sum(dim=-1)
        loss_X = loss_X.sum() / (generate_mask.sum().float() + 1e-8)
        
        loss_dict = {
            'H': loss_H,
            'X': loss_X,
        }
        
        return loss_dict

    @torch.no_grad()
    def sample_ode(self, H_init, X_init, cond_embedding, chain_ids, generate_mask, 
                   lengths, num_steps=50, method='euler', pbar=False):
        """
        Sample from the flow by solving the ODE.
        
        Args:
            H_init, X_init: initial states (typically noise)
            cond_embedding: conditional information
            chain_ids: chain identifiers
            generate_mask: generation mask
            lengths: batch lengths
            num_steps: number of integration steps
            method: integration method ('euler', 'rk4')
            pbar: show progress bar
            
        Returns:
            trajectory: dictionary with sampled states at each step
        """
        batch_ids = length_to_batch_id(lengths)
        dt = 1.0 / num_steps
        
        # Get edges
        edges, edge_types = self._get_edges(chain_ids, batch_ids, lengths)
        
        # Initialize trajectory
        traj = {0: (H_init.clone(), X_init.clone())}
        
        if pbar:
            pbar = functools.partial(tqdm, total=num_steps, desc='Flow Sampling')
        else:
            pbar = lambda x: x
        
        H_t, X_t = H_init.clone(), X_init.clone()
        
        for step in pbar(range(num_steps)):
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
            
            traj[step + 1] = (H_t.clone(), X_t.clone())
        
        return traj

    def forward(self, H_0, X_0, H_1, X_1, cond_embedding, chain_ids, generate_mask, lengths):
        """
        Forward pass for training.
        """
        return self.conditional_flow_matching_loss(
            H_0, X_0, H_1, X_1, cond_embedding, chain_ids, generate_mask, lengths
        )