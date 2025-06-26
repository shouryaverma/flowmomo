#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch_scatter import scatter_mean

from data.bioparse import VOCAB

import utils.register as R
from utils.oom_decorator import oom_decorator
from utils.nn_utils import SinusoidalPositionEmbedding
from utils.gnn_utils import length_to_batch_id, std_conserve_scatter_mean

# Import both for backward compatibility
from .flow_matching import FlowMatching  # Original
from ..IterVAE.model import CondIterAutoEncoder
from ..modules.nn import GINEConv, MLP


@R.register('RectifiedFlowMolDesign')
class RectifiedFlowMolDesign(nn.Module):
    """
    Rectified Flow-based molecular design model.
    Uses VAE + Rectified Flow with optional Contrastive Flow Matching for molecular generation.
    """

    def __init__(
            self,
            autoencoder_ckpt,
            latent_deterministic,
            hidden_size,
            h_loss_weight=None,
            std=10.0,
            is_aa_corrupt_ratio=0.1,
            flow_opt={
                'encoder_type': 'EPT',
                'encoder_opt': {'n_layers': 3},
                'sigma': 1e-4,
                'lambda_contrast': 0.05,  # Default: standard flow matching
            }
        ):
        super().__init__()
        self.latent_deterministic = latent_deterministic

        # Load frozen autoencoder
        self.autoencoder: CondIterAutoEncoder = torch.load(autoencoder_ckpt, map_location='cpu', weights_only=False)
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.autoencoder.eval()
        
        latent_size = self.autoencoder.latent_size

        # Topology embedding components
        self.bond_embed = nn.Embedding(5, hidden_size)  # [None, single, double, triple, aromatic]
        self.atom_embed = nn.Embedding(VOCAB.get_num_atom_type(), hidden_size)
        self.topo_gnn = GINEConv(hidden_size, hidden_size, hidden_size, hidden_size)

        # Positional and context embeddings
        self.position_encoding = SinusoidalPositionEmbedding(hidden_size)
        self.is_aa_embed = nn.Embedding(2, hidden_size)  # is or is not standard amino acid

        # Condition embedding MLP
        self.cond_mlp = MLP(
            input_size=3 * hidden_size,  # [position, topo, is_aa]
            hidden_size=hidden_size,
            output_size=hidden_size,
            n_layers=3,
            dropout=0.1
        )

        # Choose flow matching type based on lambda_contrast
        lambda_contrast = flow_opt.get('lambda_contrast', 0.05)
        self.use_contrastive = lambda_contrast > 0.0
        
        if self.use_contrastive:
            print(f"🚀 Using Contrastive Flow Matching with λ={lambda_contrast}")
            self.flow_matching = FlowMatching(
                latent_size=latent_size,
                hidden_size=hidden_size,
                **flow_opt
            )
        else:
            print("📋 Using standard Flow Matching")
            # Remove lambda_contrast for standard flow matching
            standard_flow_opt = {k: v for k, v in flow_opt.items() if k != 'lambda_contrast'}
            self.flow_matching = FlowMatching(
                latent_size=latent_size,
                hidden_size=hidden_size,
                **standard_flow_opt
            )
        
        # Loss weighting
        if h_loss_weight is None:
            self.h_loss_weight = 3 / latent_size  # make loss_X and loss_H about the same size
        else:
            self.h_loss_weight = float(h_loss_weight)
            
        self.register_buffer('std', torch.tensor(float(std), dtype=torch.float))
        self.is_aa_corrupt_ratio = float(is_aa_corrupt_ratio)

    @oom_decorator
    def forward(
            self,
            X,              # [Natom, 3], atom coordinates     
            S,              # [Nblock], block types
            A,              # [Natom], atom types
            bonds,          # [Nbonds, 3], chemical bonds, src-dst-type (single: 1, double: 2, triple: 3)
            position_ids,   # [Nblock], block position ids
            chain_ids,      # [Nblock], split different chains
            generate_mask,  # [Nblock], 1 for generation, 0 for context
            center_mask,    # [Nblock], 1 for used to calculate complex center of mass
            block_lengths,  # [Nblock], number of atoms in each block
            lengths,        # [batch_size]
            is_aa,          # [Nblock], 1 for amino acid (for determining the X_mask in inverse folding)
        ):
        """
        Forward pass for training the rectified flow model.
        """

        # Encode latent representations using frozen autoencoder
        with torch.no_grad():
            self.autoencoder.eval()
            Zh_1, Zx_1, _, _, _, _, _, _ = self.autoencoder.encode(
                X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths, 
                deterministic=self.latent_deterministic
            )  # [Nblock, d_latent], [Nblock, 3]

        # Normalize positions
        batch_ids = length_to_batch_id(lengths)
        Zx_1, centers = self._normalize_position(Zx_1, batch_ids, center_mask)

        # Sample noise as source distribution (H_0, X_0)
        Zh_0 = torch.randn_like(Zh_1)
        Zx_0 = torch.randn_like(Zx_1)

        # Get conditional embeddings
        position_embedding = self.position_encoding(position_ids)
        topo_embedding = self.topo_embedding(A, bonds, length_to_batch_id(block_lengths), generate_mask)

        # Amino acid embedding with corruption for robustness
        corrupt_mask = generate_mask & (torch.rand_like(is_aa, dtype=torch.float) < self.is_aa_corrupt_ratio)
        is_aa_embedding = self.is_aa_embed(
            torch.where(corrupt_mask, torch.zeros_like(is_aa), is_aa).long()
        )

        # Combine conditional information
        cond_embedding = self.cond_mlp(torch.cat([position_embedding, topo_embedding, is_aa_embedding], dim=-1))

        # Compute flow matching loss (standard or contrastive)
        loss_dict = self.flow_matching(
            H_0=Zh_0,
            X_0=Zx_0, 
            H_1=Zh_1,
            X_1=Zx_1,
            cond_embedding=cond_embedding,
            chain_ids=chain_ids,
            generate_mask=generate_mask,
            lengths=lengths
        )

        # Weight the losses
        loss_dict['total'] = loss_dict['H'] * self.h_loss_weight + loss_dict['X']

        # Log contrastive-specific metrics if available
        if self.use_contrastive and 'H_neg' in loss_dict:
            loss_dict['contrastive_strength'] = loss_dict['H_neg'] + loss_dict['X_neg']
            loss_dict['positive_strength'] = loss_dict['H_pos'] + loss_dict['X_pos']

        return loss_dict

    def topo_embedding(self, A, bonds, block_ids, generate_mask):
        """
        Generate topology embeddings from molecular structure.
        Only uses context (non-generation) parts for topology information.
        """
        ctx_mask = ~generate_mask[block_ids]

        # Only retain bonds in the context
        bond_select_mask = ctx_mask[bonds[:, 0]] & ctx_mask[bonds[:, 1]]
        bonds = bonds[bond_select_mask]

        # Embed bond and atom types
        edge_attr = self.bond_embed(bonds[:, 2])
        H = self.atom_embed(A)

        # Get topology embedding via GNN
        topo_embedding = self.topo_gnn(H, bonds[:, :2].T, edge_attr)  # [Natom]

        # Aggregate to block level
        topo_embedding = std_conserve_scatter_mean(topo_embedding, block_ids, dim=0)  # [Nblock]

        # Set generation part to zero (no topology information for generation)
        topo_embedding = torch.where(
            generate_mask[:, None].expand_as(topo_embedding),
            torch.zeros_like(topo_embedding),
            topo_embedding
        )

        return topo_embedding

    def _normalize_position(self, X, batch_ids, center_mask):
        """Normalize positions by centering and scaling."""
        centers = scatter_mean(X[center_mask], batch_ids[center_mask], dim=0, dim_size=batch_ids.max() + 1)  # [bs, 3]
        centers = centers[batch_ids]  # [N, 3]
        X = (X - centers) / self.std
        return X, centers

    def _unnormalize_position(self, X_norm, centers, batch_ids):
        """Unnormalize positions."""
        X = X_norm * self.std + centers
        return X

    @torch.no_grad()
    def sample(
            self,
            X,              # [Natom, 3], atom coordinates     
            S,              # [Nblock], block types
            A,              # [Natom], atom types
            bonds,          # [Nbonds, 3], chemical bonds, src-dst-type (single: 1, double: 2, triple: 3)
            position_ids,   # [Nblock], block position ids
            chain_ids,      # [Nblock], split different chains
            generate_mask,  # [Nblock], 1 for generation, 0 for context
            center_mask,    # [Nblock], 1 for calculating complex mass center
            block_lengths,  # [Nblock], number of atoms in each block
            lengths,        # [batch_size]
            is_aa,          # [Nblock], 1 for amino acid (for determining the X_mask in inverse folding)
            sample_opt={
                'num_steps': 50,
                'method': 'euler',  # 'euler' or 'rk4'
                'pbar': False,
                'vae_decode_n_iter': 10,
            },
            return_tensor=False,
        ):
        """
        Sample molecular structures using rectified flow.
        """

        vae_decode_n_iter = sample_opt.pop('vae_decode_n_iter', 10)
        flow_sample_opt = {k: v for k, v in sample_opt.items() if k != 'vae_decode_n_iter'}

        block_ids = length_to_batch_id(block_lengths)

        # Ensure no data leakage from generation parts
        S[generate_mask] = 0
        X[generate_mask[block_ids]] = 0
        A[generate_mask[block_ids]] = 0
        ctx_atom_mask = ~generate_mask[block_ids]
        bonds = bonds[ctx_atom_mask[bonds[:, 0]] & ctx_atom_mask[bonds[:, 1]]]

        # Encode context using autoencoder
        self.autoencoder.eval()
        Zh_ctx, Zx_ctx, _, _, _, _, _, _ = self.autoencoder.encode(
            X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths, 
            deterministic=self.latent_deterministic
        )

        # Normalize positions
        batch_ids = length_to_batch_id(lengths)
        Zx_ctx, centers = self._normalize_position(Zx_ctx, batch_ids, center_mask)

        # Initialize with noise for generation parts
        Zh_init = torch.where(
            generate_mask[:, None].expand_as(Zh_ctx),
            torch.randn_like(Zh_ctx),
            Zh_ctx
        )
        Zx_init = torch.where(
            generate_mask[:, None].expand_as(Zx_ctx),
            torch.randn_like(Zx_ctx),
            Zx_ctx
        )

        # Get conditional embeddings
        topo_embedding = self.topo_embedding(A, bonds, length_to_batch_id(block_lengths), generate_mask)
        position_embedding = self.position_encoding(position_ids)
        is_aa_embedding = self.is_aa_embed(is_aa.long())
        cond_embedding = self.cond_mlp(torch.cat([position_embedding, topo_embedding, is_aa_embedding], dim=-1))
        
        # Sample using flow matching (works for both standard and contrastive)
        traj = self.flow_matching.sample_ode(
            H_init=Zh_init,
            X_init=Zx_init,
            cond_embedding=cond_embedding,
            chain_ids=chain_ids,
            generate_mask=generate_mask,
            lengths=lengths,
            **flow_sample_opt
        )
        
        # Get final sampled states
        final_step = max(traj.keys())
        H_final, X_final = traj[final_step]
        
        # Ensure context parts remain unchanged
        H_final = torch.where(generate_mask[:, None].expand_as(H_final), H_final, Zh_ctx)
        X_final = torch.where(generate_mask[:, None].expand_as(X_final), X_final, Zx_ctx)

        # Unnormalize positions
        X_final = self._unnormalize_position(X_final, centers, batch_ids)

        # Decode using autoencoder
        return self.autoencoder.generate(
            X=X, S=S, A=A, bonds=bonds, position_ids=position_ids,
            chain_ids=chain_ids, generate_mask=generate_mask, block_lengths=block_lengths,
            lengths=lengths, is_aa=is_aa, given_latent=(H_final, X_final, None),
            n_iter=vae_decode_n_iter, topo_generate_mask=generate_mask
        )

    def get_trajectory(self, *args, **kwargs):
        """
        Get the full sampling trajectory for analysis.
        Similar to sample() but returns the full trajectory.
        """
        # Remove return_tensor and vae_decode_n_iter from kwargs for flow sampling
        sample_opt = kwargs.pop('sample_opt', {})
        flow_sample_opt = {k: v for k, v in sample_opt.items() if k not in ['vae_decode_n_iter', 'return_tensor']}
        
        # Get all the arguments from sample method
        (X, S, A, bonds, position_ids, chain_ids, generate_mask, center_mask, 
         block_lengths, lengths, is_aa) = args
        
        block_ids = length_to_batch_id(block_lengths)
        
        # Ensure no data leakage
        S[generate_mask] = 0
        X[generate_mask[block_ids]] = 0
        A[generate_mask[block_ids]] = 0
        ctx_atom_mask = ~generate_mask[block_ids]
        bonds = bonds[ctx_atom_mask[bonds[:, 0]] & ctx_atom_mask[bonds[:, 1]]]

        # Encode context
        self.autoencoder.eval()
        Zh_ctx, Zx_ctx, _, _, _, _, _, _ = self.autoencoder.encode(
            X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths, 
            deterministic=self.latent_deterministic
        )

        # Normalize and prepare
        batch_ids = length_to_batch_id(lengths)
        Zx_ctx, centers = self._normalize_position(Zx_ctx, batch_ids, center_mask)
        
        Zh_init = torch.where(generate_mask[:, None].expand_as(Zh_ctx), torch.randn_like(Zh_ctx), Zh_ctx)
        Zx_init = torch.where(generate_mask[:, None].expand_as(Zx_ctx), torch.randn_like(Zx_ctx), Zx_ctx)

        # Get embeddings
        topo_embedding = self.topo_embedding(A, bonds, length_to_batch_id(block_lengths), generate_mask)
        position_embedding = self.position_encoding(position_ids)
        is_aa_embedding = self.is_aa_embed(is_aa.long())
        cond_embedding = self.cond_mlp(torch.cat([position_embedding, topo_embedding, is_aa_embedding], dim=-1))
        
        # Get full trajectory
        traj = self.flow_matching.sample_ode(
            H_init=Zh_init,
            X_init=Zx_init,
            cond_embedding=cond_embedding,
            chain_ids=chain_ids,
            generate_mask=generate_mask,
            lengths=lengths,
            **flow_sample_opt
        )
        
        # Unnormalize all trajectory points
        for step in traj:
            H_step, X_step = traj[step]
            X_step = self._unnormalize_position(X_step, centers, batch_ids)
            traj[step] = (H_step, X_step)
        
        return traj