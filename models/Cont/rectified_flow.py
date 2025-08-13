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

# Import FIXED flow matching
from .flow_matching import FlowMatching
from ..IterVAE.model import CondIterAutoEncoder
from ..modules.nn import GINEConv, MLP


@R.register('RectifiedFlowMolDesign')
class RectifiedFlowMolDesign(nn.Module):
    """
    FIXED Rectified Flow-based molecular design model.
    Now works directly on IterVAE latents without LDM.
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
                'p_uncond': 0.1,
                'use_ot_coupling': True,  # FIXED: Always enabled
                'time_weighting': 'uniform',  # FIXED: Start simple
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

        # Topology embedding components (same as LDM)
        self.bond_embed = nn.Embedding(5, hidden_size)
        self.atom_embed = nn.Embedding(VOCAB.get_num_atom_type(), hidden_size)
        self.topo_gnn = GINEConv(hidden_size, hidden_size, hidden_size, hidden_size)

        # Positional and context embeddings (same as LDM)
        self.position_encoding = SinusoidalPositionEmbedding(hidden_size)
        self.is_aa_embed = nn.Embedding(2, hidden_size)

        # Condition embedding MLP (same as LDM)
        self.cond_mlp = MLP(
            input_size=3 * hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            n_layers=3,
            dropout=0.1
        )

        # FIXED: Initialize flow matching with simplified settings
        self.flow_matching = FlowMatching(
            latent_size=latent_size,
            hidden_size=hidden_size,
            **flow_opt
        )
        
        # Loss weighting (same as LDM)
        if h_loss_weight is None:
            self.h_loss_weight = 3 / latent_size
        else:
            self.h_loss_weight = float(h_loss_weight)
            
        self.register_buffer('std', torch.tensor(float(std), dtype=torch.float))
        self.is_aa_corrupt_ratio = float(is_aa_corrupt_ratio)

    @oom_decorator
    def forward(self, X, S, A, bonds, position_ids, chain_ids, generate_mask, center_mask, block_lengths, lengths, is_aa):
        """
        FIXED forward pass - much simpler now
        """
        
        # Encode using frozen autoencoder (same as LDM)
        with torch.no_grad():
            self.autoencoder.eval()
            Zh_1, Zx_1, _, _, _, _, _, _ = self.autoencoder.encode(
                X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths,
                deterministic=self.latent_deterministic
            )

        # Normalize positions (same as LDM)
        batch_ids = length_to_batch_id(lengths)
        Zx_1, centers = self._normalize_position(Zx_1, batch_ids, center_mask)

        # Get conditional embeddings (same as LDM)
        position_embedding = self.position_encoding(position_ids)
        topo_embedding = self.topo_embedding(A, bonds, length_to_batch_id(block_lengths), generate_mask)

        # Amino acid embedding with corruption (same as LDM)
        corrupt_mask = generate_mask & (torch.rand_like(is_aa, dtype=torch.float) < self.is_aa_corrupt_ratio)
        is_aa_embedding = self.is_aa_embed(
            torch.where(corrupt_mask, torch.zeros_like(is_aa), is_aa).long()
        )

        # Combine conditional information (same as LDM)
        cond_embedding = self.cond_mlp(torch.cat([position_embedding, topo_embedding, is_aa_embedding], dim=-1))

        # FIXED: Flow matching handles everything internally now
        loss_dict = self.flow_matching.contrastive_flow_matching_loss(
            H_0=None, X_0=None, H_1=Zh_1, X_1=Zx_1,  # H_0, X_0 will be sampled internally
            cond_embedding=cond_embedding, chain_ids=chain_ids,
            generate_mask=generate_mask, lengths=lengths
        )

        # Same loss weighting as LDM
        loss_dict['total'] = loss_dict['H'] * self.h_loss_weight + loss_dict['X']

        return loss_dict

    def topo_embedding(self, A, bonds, block_ids, generate_mask):
        """
        Generate topology embeddings (same as LDM)
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

        # Set generation part to zero
        topo_embedding = torch.where(
            generate_mask[:, None].expand_as(topo_embedding),
            torch.zeros_like(topo_embedding),
            topo_embedding
        )

        return topo_embedding

    def _normalize_position(self, X, batch_ids, center_mask):
        """Normalize positions (same as LDM)"""
        centers = scatter_mean(X[center_mask], batch_ids[center_mask], dim=0, dim_size=batch_ids.max() + 1)
        centers = centers[batch_ids]
        X = (X - centers) / self.std
        return X, centers

    def _unnormalize_position(self, X_norm, centers, batch_ids):
        """Unnormalize positions (same as LDM)"""
        X = X_norm * self.std + centers
        return X

    @torch.no_grad()
    def sample(self, X, S, A, bonds, position_ids, chain_ids, generate_mask, center_mask, block_lengths, lengths, is_aa, sample_opt={}, return_tensor=False):
        """
        FIXED sampling - much simpler and more reliable
        """
        
        vae_decode_n_iter = sample_opt.pop('vae_decode_n_iter', 10)
        guidance_scale = sample_opt.pop('guidance_scale', 1.0)
        flow_sample_opt = {k: v for k, v in sample_opt.items() if k not in ['vae_decode_n_iter', 'guidance_scale']}

        block_ids = length_to_batch_id(block_lengths)

        # Data preparation (same as LDM)
        S[generate_mask] = 0
        X[generate_mask[block_ids]] = 0
        A[generate_mask[block_ids]] = 0
        ctx_atom_mask = ~generate_mask[block_ids]
        bonds = bonds[ctx_atom_mask[bonds[:, 0]] & ctx_atom_mask[bonds[:, 1]]]

        # Encode context (same as LDM)
        self.autoencoder.eval()
        Zh_ctx, Zx_ctx, _, _, _, _, _, _ = self.autoencoder.encode(
            X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths,
            deterministic=self.latent_deterministic
        )

        # Normalize positions (same as LDM)
        batch_ids = length_to_batch_id(lengths)
        Zx_ctx, centers = self._normalize_position(Zx_ctx, batch_ids, center_mask)

        # FIXED: Simple prior sampling based on molecular size
        if generate_mask.sum() > 0:
            gen_indices = torch.nonzero(generate_mask, as_tuple=False).squeeze(-1)
            
            # Sample priors for generated parts
            Zh_gen_init, Zx_gen_init = self.flow_matching.sample_molecular_prior(
                Zh_ctx[gen_indices].shape, Zx_ctx[gen_indices].shape, Zh_ctx.device, block_lengths
            )
            
            # Combine with context
            Zh_init = Zh_ctx.clone()
            Zx_init = Zx_ctx.clone()
            Zh_init[gen_indices] = Zh_gen_init
            Zx_init[gen_indices] = Zx_gen_init
        else:
            Zh_init = Zh_ctx
            Zx_init = Zx_ctx

        # Get conditional embeddings (same as LDM)
        topo_embedding = self.topo_embedding(A, bonds, length_to_batch_id(block_lengths), generate_mask)
        position_embedding = self.position_encoding(position_ids)
        is_aa_embedding = self.is_aa_embed(is_aa.long())
        cond_embedding = self.cond_mlp(torch.cat([position_embedding, topo_embedding, is_aa_embedding], dim=-1))
        
        # FIXED: Sample using flow matching
        traj = self.flow_matching.sample_ode(
            H_init=Zh_init, X_init=Zx_init, cond_embedding=cond_embedding,
            chain_ids=chain_ids, generate_mask=generate_mask, lengths=lengths,
            guidance_scale=guidance_scale,
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

        # Decode using autoencoder (same as LDM)
        return self.autoencoder.generate(
            X=X, S=S, A=A, bonds=bonds, position_ids=position_ids,
            chain_ids=chain_ids, generate_mask=generate_mask, block_lengths=block_lengths,
            lengths=lengths, is_aa=is_aa, given_latent=(H_final, X_final, None),
            n_iter=vae_decode_n_iter, topo_generate_mask=generate_mask
        )

    def get_trajectory(self, *args, **kwargs):
        """
        FIXED trajectory sampling
        """
        # Remove special kwargs
        sample_opt = kwargs.pop('sample_opt', {})
        guidance_scale = sample_opt.pop('guidance_scale', 1.0)
        flow_sample_opt = {k: v for k, v in sample_opt.items() if k not in ['vae_decode_n_iter', 'return_tensor', 'guidance_scale']}
        
        # Get arguments
        (X, S, A, bonds, position_ids, chain_ids, generate_mask, center_mask, 
        block_lengths, lengths, is_aa) = args
        
        block_ids = length_to_batch_id(block_lengths)
        
        # Data preparation
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
        
        # Sample initial states
        if generate_mask.sum() > 0:
            gen_indices = torch.nonzero(generate_mask, as_tuple=False).squeeze(-1)
            
            Zh_gen_init, Zx_gen_init = self.flow_matching.sample_molecular_prior(
                Zh_ctx[gen_indices].shape, Zx_ctx[gen_indices].shape, Zh_ctx.device, block_lengths
            )
            
            Zh_init = Zh_ctx.clone()
            Zx_init = Zx_ctx.clone()
            Zh_init[gen_indices] = Zh_gen_init
            Zx_init[gen_indices] = Zx_gen_init
        else:
            Zh_init = Zh_ctx
            Zx_init = Zx_ctx

        # Get embeddings
        topo_embedding = self.topo_embedding(A, bonds, length_to_batch_id(block_lengths), generate_mask)
        position_embedding = self.position_encoding(position_ids)
        is_aa_embedding = self.is_aa_embed(is_aa.long())
        cond_embedding = self.cond_mlp(torch.cat([position_embedding, topo_embedding, is_aa_embedding], dim=-1))
        
        # Get trajectory
        traj = self.flow_matching.sample_ode(
            H_init=Zh_init,
            X_init=Zx_init,
            cond_embedding=cond_embedding,
            chain_ids=chain_ids,
            generate_mask=generate_mask,
            lengths=lengths,
            guidance_scale=guidance_scale,
            **flow_sample_opt
        )
        
        # Unnormalize all trajectory points
        for step in traj:
            H_step, X_step = traj[step]
            X_step = self._unnormalize_position(X_step, centers, batch_ids)
            traj[step] = (H_step, X_step)
        
        return traj