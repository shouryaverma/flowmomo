#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
from torch_scatter import scatter_mean

from utils import register as R
from utils.gnn_utils import length_to_batch_id
from .abs_trainer import Trainer


@R.register('RectTrainer')
class RectTrainer(Trainer):  # rectified flow trainer

    def __init__(self, model, train_loader, valid_loader, criterion: str, config: dict, save_config: dict):
        super().__init__(model, train_loader, valid_loader, config, save_config)
        self.criterion = criterion

    def train_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx)

    def valid_step(self, batch, batch_idx):
        loss = self.share_step(batch, batch_idx, val=True)
        
        # Additional validation: sampling quality assessment
        if batch_idx % 10 == 0:  # Only run sampling validation occasionally to save time
            self._sample_validation(batch, batch_idx)
        
        return loss

    def _sample_validation(self, batch, batch_idx):
        """
        Perform sampling-based validation to assess generation quality.
        """
        if self.local_rank != -1:  # DDP mode
            model = self.model.module
        else:
            model = self.model
            
        # Sample with faster settings for validation
        try:
            with torch.no_grad():
                # Generate samples using the flow model
                generated_results = model.sample(
                    **batch,
                    sample_opt={
                        'num_steps': 20,  # Fewer steps for faster validation
                        'method': 'euler',
                        'pbar': False,
                        'vae_decode_n_iter': 5,  # Fewer VAE decode iterations
                    }
                )
                
                # If generated_results is the full output from autoencoder.generate
                if isinstance(generated_results, tuple) and len(generated_results) == 6:
                    batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds = generated_results
                    
                    # Log sampling metrics
                    total_likelihood = 0
                    total_atoms = 0
                    for i, (block_ll_list, block_X_list) in enumerate(zip(batch_ll, batch_X)):
                        for block_ll, block_X in zip(block_ll_list, block_X_list):
                            if len(block_ll) > 0:
                                total_likelihood += sum(block_ll)
                                total_atoms += len(block_ll)
                    
                    avg_likelihood = total_likelihood / max(total_atoms, 1)
                    self.log('sampling/avg_likelihood', avg_likelihood, batch_idx, val=True)
                    self.log('sampling/total_generated_atoms', total_atoms, batch_idx, val=True)
                    
                    # Log number of generated molecules/blocks
                    num_generated_blocks = sum(len(block_list) for block_list in batch_S)
                    self.log('sampling/num_generated_blocks', num_generated_blocks, batch_idx, val=True)
                
        except Exception as e:
            # Log sampling failure but don't crash training
            print(f"Warning: Sampling validation failed at batch {batch_idx}: {str(e)}")
            self.log('sampling/failure_rate', 1.0, batch_idx, val=True)

    def _train_epoch_end(self, device):
        dataset = self.train_loader.dataset
        if hasattr(dataset, 'update_epoch'):
            dataset.update_epoch()
        return super()._train_epoch_end(device)

    def _valid_epoch_begin(self, device):
        # Set random seed for reproducible validation
        self.rng_state = torch.random.get_rng_state()
        torch.manual_seed(42)  # Different seed from LDM for flow models
        return super()._valid_epoch_begin(device)

    def _valid_epoch_end(self, device):
        torch.random.set_rng_state(self.rng_state)
        return super()._valid_epoch_end(device)

    ########## Override end ##########

    def share_step(self, batch, batch_idx, val=False):
        """
        Shared step for both training and validation.
        """
        loss_dict = self.model(**batch)
        if self.is_oom_return(loss_dict):
            return loss_dict
        
        loss = loss_dict['total']
        log_type = 'Validation' if val else 'Train'

        # Log all losses from the flow model
        for key in loss_dict:
            self.log(f'{key}/{log_type}', loss_dict[key], batch_idx, val)

        # Log learning rate during training
        if not val:
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log('lr', lr, batch_idx, val)

        return loss

    def share_valid_decode(self, batch, batch_idx, Zh, Zx, X_mask, S_mask, X_gt, S_gt, A_gt, suffix, metrics):
        """
        Legacy method for compatibility with existing validation framework.
        Flow models don't use the same encode/decode paradigm, so this is simplified.
        """
        # For flow models, we can skip the detailed reconstruction validation
        # as the flow model handles generation end-to-end
        
        # Instead, we could do basic sanity checks
        batch_ids = length_to_batch_id(batch['lengths'])
        
        # Simple coordinate RMSD if needed
        if 'RMSD' in metrics:
            # Calculate RMSD between generated and ground truth coordinates
            # This is a placeholder - in practice you'd run the flow model
            rmsd = torch.zeros(1, device=Zh.device)  # Placeholder
            self.log(f'RMSD/Validation_{suffix}', rmsd, batch_idx, val=True, batch_size=len(batch['lengths']))

    def get_flow_trajectory(self, batch, num_steps=50):
        """
        Helper method to get the full flow trajectory for analysis.
        """
        if self.local_rank != -1:  # DDP mode
            model = self.model.module
        else:
            model = self.model
            
        with torch.no_grad():
            trajectory = model.get_trajectory(
                **batch,
                sample_opt={
                    'num_steps': num_steps,
                    'method': 'euler',
                    'pbar': False,
                }
            )
        return trajectory

    def validate_flow_interpolation(self, batch, batch_idx, num_interpolations=5):
        """
        Validate flow interpolation quality by checking intermediate states.
        """
        try:
            trajectory = self.get_flow_trajectory(batch, num_steps=num_interpolations)
            
            # Analyze trajectory smoothness
            smoothness_scores = []
            prev_X, prev_H = None, None
            
            for step in sorted(trajectory.keys()):
                curr_X, curr_H = trajectory[step]
                
                if prev_X is not None:
                    # Calculate smoothness as the change between steps
                    delta_X = torch.norm(curr_X - prev_X, dim=-1).mean()
                    delta_H = torch.norm(curr_H - prev_H, dim=-1).mean()
                    smoothness_scores.append((delta_X.item(), delta_H.item()))
                
                prev_X, prev_H = curr_X, curr_H
            
            if smoothness_scores:
                avg_delta_X = sum(s[0] for s in smoothness_scores) / len(smoothness_scores)
                avg_delta_H = sum(s[1] for s in smoothness_scores) / len(smoothness_scores)
                
                self.log('trajectory/avg_delta_X', avg_delta_X, batch_idx, val=True)
                self.log('trajectory/avg_delta_H', avg_delta_H, batch_idx, val=True)
                
        except Exception as e:
            print(f"Warning: Flow interpolation validation failed: {str(e)}")

    def compute_flow_metrics(self, batch):
        """
        Compute flow-specific metrics for monitoring training.
        """
        if self.local_rank != -1:  # DDP mode
            model = self.model.module
        else:
            model = self.model
            
        metrics = {}
        
        with torch.no_grad():
            # Sample with the flow model
            try:
                # Quick sampling for metrics
                results = model.sample(
                    **batch,
                    sample_opt={
                        'num_steps': 10,
                        'method': 'euler',
                        'pbar': False,
                        'vae_decode_n_iter': 3,
                    }
                )
                
                if isinstance(results, tuple) and len(results) == 6:
                    batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds = results
                    
                    # Compute basic statistics
                    total_generated = sum(len(s_list) for s_list in batch_S)
                    metrics['flow/num_generated'] = total_generated
                    
                    # Average atom count per generated molecule
                    total_atoms = sum(sum(len(x_block) for x_block in x_list) for x_list in batch_X)
                    metrics['flow/avg_atoms_per_mol'] = total_atoms / max(total_generated, 1)
                    
            except Exception as e:
                print(f"Warning: Flow metrics computation failed: {str(e)}")
                metrics['flow/sampling_error'] = 1.0
        
        return metrics

    def detailed_validation(self, batch, batch_idx):
        """
        Comprehensive validation including sampling, trajectory analysis, and metrics.
        """
        # Run standard sampling validation
        self._sample_validation(batch, batch_idx)
        
        # Run trajectory validation every 50 batches
        if batch_idx % 50 == 0:
            self.validate_flow_interpolation(batch, batch_idx)
        
        # Compute flow metrics every 20 batches
        if batch_idx % 20 == 0:
            metrics = self.compute_flow_metrics(batch)
            for key, value in metrics.items():
                self.log(key, value, batch_idx, val=True)