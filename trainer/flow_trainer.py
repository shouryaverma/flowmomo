#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import numpy as np
from torch_scatter import scatter_mean
from utils import register as R
from utils.gnn_utils import length_to_batch_id
from utils.chem_utils import (
    check_stability, diversity, similarity, mol2smi, smi2mol,
    calculate_1dqsar_repr, AtomVocab
)
from .abs_trainer import Trainer


@R.register('FlowTrainer')
class FlowTrainer(Trainer):
    def __init__(self, model, train_loader, valid_loader, criterion: str, config: dict, save_config: dict):
        super().__init__(model, train_loader, valid_loader, config, save_config)
        self.criterion = criterion
        self.atom_vocab = AtomVocab()
        
        # Validation settings
        self.val_sample_steps = config.get('val_sample_steps', 20)
        self.val_apply_constraints = config.get('val_apply_constraints', True)
        
    def train_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx)
        
    def valid_step(self, batch, batch_idx):
        loss = self.share_step(batch, batch_idx, val=True)
       
        # Flow-specific validation metrics
        if self.local_rank != -1:
            model = self.model.module
        else:
            model = self.model
           
        # Sample and compute molecular quality metrics
        with torch.no_grad():
            try:
                generated_results = model.sample(
                    X=batch['X'], S=batch['S'], A=batch['A'],
                    bonds=batch['bonds'], position_ids=batch['position_ids'],
                    chain_ids=batch['chain_ids'], generate_mask=batch['generate_mask'],
                    block_lengths=batch['block_lengths'], lengths=batch['lengths'],
                    is_aa=batch['is_aa'], num_steps=self.val_sample_steps,
                    apply_constraints=self.val_apply_constraints
                )
               
                if isinstance(generated_results, tuple) and len(generated_results) >= 3:
                    batch_S, batch_X, batch_A = generated_results[:3]
                    
                    # Compute comprehensive molecular validation metrics
                    metrics = self.compute_molecular_metrics(
                        batch_S, batch_X, batch_A, batch
                    )
                    
                    # Log all metrics
                    for metric_name, metric_value in metrics.items():
                        self.log(f'{metric_name}/Validation', metric_value, batch_idx, val=True)
                   
            except Exception as e:
                # If sampling fails, log failure
                self.log('validation_sampling_success', 0.0, batch_idx, val=True)
                print(f"Validation sampling failed: {e}")
               
        return loss

    def share_step(self, batch, batch_idx, val=False):
        loss_dict = self.model(**batch)
        if self.is_oom_return(loss_dict):
            return loss_dict
           
        log_type = 'Validation' if val else 'Train'
        for key in loss_dict:
            self.log(f'{key}/{log_type}', loss_dict[key], batch_idx, val)
            
        if not val:
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log('lr', lr, batch_idx, val)
            
        return loss_dict['total']

    def compute_molecular_metrics(self, batch_S, batch_X, batch_A, original_batch):
        """Compute comprehensive molecular quality metrics"""
        metrics = {}
        
        # Basic sampling success rate
        valid_samples = sum(1 for s in batch_S if len(s) > 0)
        total_samples = len(batch_S)
        metrics['sampling_success_rate'] = valid_samples / max(total_samples, 1)
        
        if valid_samples == 0:
            # If no valid samples, return minimal metrics
            for metric in ['rmsd', 'validity', 'stability', 'diversity', 'uniqueness']:
                metrics[metric] = 0.0
            return metrics
        
        # Process valid samples
        valid_coords, valid_atoms, true_coords, true_atoms = [], [], [], []
        
        for i, (s, x_list, a_list) in enumerate(zip(batch_S, batch_X, batch_A)):
            if len(s) > 0 and len(x_list) > 0 and len(a_list) > 0:
                try:
                    # Generated coordinates and atoms
                    gen_coords = torch.tensor(x_list[0], dtype=torch.float32)
                    gen_atoms = torch.tensor(a_list[0], dtype=torch.long)
                    
                    if len(gen_coords) > 0:
                        valid_coords.append(gen_coords)
                        valid_atoms.append(gen_atoms)
                        
                        # Corresponding true coordinates (from original batch)
                        # This is simplified - in practice you'd need to match generation regions
                        batch_size = len(batch_S)
                        if i < len(original_batch['X']):
                            true_coords.append(original_batch['X'])
                            true_atoms.append(original_batch['A'])
                            
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    continue
        
        if len(valid_coords) == 0:
            for metric in ['rmsd', 'validity', 'stability', 'diversity', 'uniqueness']:
                metrics[metric] = 0.0
            return metrics
        
        # 1. RMSD computation (coordinate accuracy)
        try:
            rmsd_values = []
            for gen_coords, true_coords_batch in zip(valid_coords, true_coords):
                if len(gen_coords) == len(true_coords_batch):
                    rmsd = torch.sqrt(torch.mean((gen_coords - true_coords_batch) ** 2))
                    rmsd_values.append(rmsd.item())
            
            metrics['rmsd'] = np.mean(rmsd_values) if rmsd_values else 0.0
        except Exception:
            metrics['rmsd'] = 0.0
        
        # 2. Chemical validity (proper valence, etc.)
        try:
            validity_scores = []
            for coords, atoms in zip(valid_coords, valid_atoms):
                coords_np = coords.detach().cpu().numpy()
                atoms_np = atoms.detach().cpu().numpy()
                
                # Convert atom indices to types for stability check
                atom_types = atoms_np + 1  # Assuming 0-indexed atoms
                
                is_stable, stable_bonds, total_atoms = check_stability(
                    coords_np, atom_types, return_nr_bonds=True
                )
                validity_score = stable_bonds / max(total_atoms, 1)
                validity_scores.append(validity_score)
            
            metrics['validity'] = np.mean(validity_scores) if validity_scores else 0.0
        except Exception:
            metrics['validity'] = 0.0
        
        # 3. Structural stability
        try:
            stability_scores = []
            for coords, atoms in zip(valid_coords, valid_atoms):
                coords_np = coords.detach().cpu().numpy()
                atoms_np = atoms.detach().cpu().numpy()
                atom_types = atoms_np + 1
                
                is_stable, _, _ = check_stability(coords_np, atom_types)
                stability_scores.append(float(is_stable))
            
            metrics['stability'] = np.mean(stability_scores) if stability_scores else 0.0
        except Exception:
            metrics['stability'] = 0.0
        
        # 4. Diversity (structural variety)
        try:
            if len(valid_coords) > 1:
                # Simple diversity based on coordinate differences
                diversity_scores = []
                for i in range(len(valid_coords)):
                    for j in range(i + 1, len(valid_coords)):
                        coords1, coords2 = valid_coords[i], valid_coords[j]
                        if len(coords1) == len(coords2):
                            diff = torch.mean((coords1 - coords2) ** 2)
                            diversity_scores.append(diff.item())
                
                metrics['diversity'] = np.mean(diversity_scores) if diversity_scores else 0.0
            else:
                metrics['diversity'] = 0.0
        except Exception:
            metrics['diversity'] = 0.0
        
        # 5. Uniqueness (non-redundancy)
        try:
            if len(valid_coords) > 1:
                unique_structures = []
                for coords in valid_coords:
                    # Simple uniqueness check based on coordinate hashing
                    coord_hash = hash(tuple(coords.flatten().tolist()))
                    unique_structures.append(coord_hash)
                
                metrics['uniqueness'] = len(set(unique_structures)) / len(unique_structures)
            else:
                metrics['uniqueness'] = 1.0
        except Exception:
            metrics['uniqueness'] = 1.0
        
        # 6. Molecular properties (if we can convert to SMILES)
        try:
            qed_scores = []
            for coords, atoms in zip(valid_coords, valid_atoms):
                try:
                    # This would require converting coordinates + atoms to SMILES
                    # For now, just compute basic 1D descriptors
                    coords_np = coords.detach().cpu().numpy()
                    atoms_np = atoms.detach().cpu().numpy()
                    
                    # Mock QED calculation - replace with actual implementation
                    qed_score = 0.5  # Placeholder
                    qed_scores.append(qed_score)
                    
                except Exception:
                    continue
            
            metrics['qed'] = np.mean(qed_scores) if qed_scores else 0.0
        except Exception:
            metrics['qed'] = 0.0
        
        return metrics

    def compute_rmsd(self, generated_coords, true_coords):
        """Compute RMSD between generated and true coordinates"""
        try:
            if len(generated_coords) != len(true_coords):
                return float('inf')
            
            gen_tensor = torch.tensor(generated_coords, dtype=torch.float32)
            true_tensor = torch.tensor(true_coords, dtype=torch.float32)
            
            # Align centroids
            gen_centered = gen_tensor - gen_tensor.mean(dim=0)
            true_centered = true_tensor - true_tensor.mean(dim=0)
            
            # Compute RMSD
            rmsd = torch.sqrt(torch.mean((gen_centered - true_centered) ** 2))
            return rmsd.item()
            
        except Exception:
            return float('inf')

    def assess_chemical_validity(self, coords, atoms):
        """Assess chemical validity of a generated structure"""
        try:
            coords_np = np.array(coords)
            atoms_np = np.array(atoms)
            
            # Convert to types expected by check_stability
            atom_types = atoms_np + 1  # Assuming 0-indexed
            
            is_stable, stable_bonds, total_atoms = check_stability(
                coords_np, atom_types, return_nr_bonds=True
            )
            
            return {
                'is_stable': is_stable,
                'bond_validity': stable_bonds / max(total_atoms, 1),
                'total_atoms': total_atoms
            }
            
        except Exception as e:
            return {
                'is_stable': False,
                'bond_validity': 0.0,
                'total_atoms': 0
            }