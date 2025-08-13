#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
from torch_scatter import scatter_mean

from utils import register as R
from utils.gnn_utils import length_to_batch_id
from .abs_trainer import Trainer


@R.register('RectTrainer')
class RectTrainer(Trainer):  # FIXED rectified flow trainer

    def __init__(self, model, train_loader, valid_loader, criterion: str, config: dict, save_config: dict):
        super().__init__(model, train_loader, valid_loader, config, save_config)
        self.criterion = criterion

    def train_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx)

    def valid_step(self, batch, batch_idx):
        loss = self.share_step(batch, batch_idx, val=True)
        
        # FIXED: Simplified validation - only occasionally run sampling
        if batch_idx % 20 == 0:  # Less frequent to avoid slowdown
            self._sample_validation(batch, batch_idx)
        
        return loss

    def _sample_validation(self, batch, batch_idx):
        """
        FIXED: Simplified sampling validation
        """
        if self.local_rank != -1:  # DDP mode
            model = self.model.module
        else:
            model = self.model
            
        try:
            with torch.no_grad():
                # Quick sampling for validation
                generated_results = model.sample(
                    **batch,
                    sample_opt={
                        'num_steps': 10,  # Very few steps for speed
                        'method': 'euler',
                        'pbar': False,
                        'vae_decode_n_iter': 3,  # Minimal VAE iterations
                    }
                )
                
                # Log basic metrics
                if isinstance(generated_results, tuple) and len(generated_results) == 6:
                    batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds = generated_results
                    
                    # Count generated blocks
                    num_generated_blocks = sum(len(block_list) for block_list in batch_S)
                    self.log('sampling/num_generated_blocks', num_generated_blocks, batch_idx, val=True)
                    
                    # Count generated atoms
                    total_atoms = sum(sum(len(x_block) for x_block in x_list) for x_list in batch_X)
                    self.log('sampling/total_generated_atoms', total_atoms, batch_idx, val=True)
                    
                    # Success rate
                    self.log('sampling/success_rate', 1.0, batch_idx, val=True)
                
        except Exception as e:
            # Log failure but don't crash
            print(f"Sampling validation failed at batch {batch_idx}: {str(e)}")
            self.log('sampling/success_rate', 0.0, batch_idx, val=True)

    def _train_epoch_end(self, device):
        dataset = self.train_loader.dataset
        if hasattr(dataset, 'update_epoch'):
            dataset.update_epoch()
        return super()._train_epoch_end(device)

    def _valid_epoch_begin(self, device):
        # Set reproducible random seed
        self.rng_state = torch.random.get_rng_state()
        torch.manual_seed(42)
        return super()._valid_epoch_begin(device)

    def _valid_epoch_end(self, device):
        torch.random.set_rng_state(self.rng_state)
        return super()._valid_epoch_end(device)

    def share_step(self, batch, batch_idx, val=False):
        """
        FIXED: Simplified training step
        """
        loss_dict = self.model(**batch)
        if self.is_oom_return(loss_dict):
            return loss_dict
        
        loss = loss_dict['total']
        log_type = 'Validation' if val else 'Train'

        # Log all losses
        for key in loss_dict:
            self.log(f'{key}/{log_type}', loss_dict[key], batch_idx, val)

        # Log learning rate during training
        if not val:
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log('lr', lr, batch_idx, val)

        return loss