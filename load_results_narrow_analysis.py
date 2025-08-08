#!/usr/bin/env python3
"""
Adapted script to analyze results from run_narrow.sh specifically.
Computes mean ± standard deviation across 10 random seeds for each dataset
and generates forecast plots for the specific hyperparameter combination used.

This script supports both snapMMD and CDE forecast methods and is based on 
load_results_extensive.py but specifically filters for:
- Predictors: dt_ridge_sinusoidal
- Samplers: dt_equals_one  
- Predictor loss weights: 0
- Seeds: 0-9 (10 seeds each)
- Datasets: GoM, PBMC, LV, Repressilator

For snapMMD: Loads pre-computed forecasts from snapMMD_forecasts/ directory
For CDE: Generates forecasts from trained model checkpoints
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import os
import sys
import argparse
import hydra
import logging
import yaml
from snapMMD.dls import MMDLoss, RBF
from scipy.optimize import linprog
from sklearn.decomposition import PCA
import plotly.graph_objects as go

# Add imports needed for CDE mode
from utils.experiment_utils import load_best_model, get_experiment_info
from latent_mapping_training import load_latent_mapping_model

# Import the existing UnifiedResultsLoader
from load_results_extensive import (
    UnifiedResultsLoader, 
    DATASET_CONFIGS, 
    calculate_mmd_scores, 
    calculate_emd_scores,
    calculate_emd
)

class NarrowAnalysisLoader(UnifiedResultsLoader):
    """
    Specialized loader for analyzing the specific results from run_narrow.sh.
    Filters for the exact hyperparameter combination and computes statistics across seeds.
    """
    
    def __init__(self, dataset_name, forecast_method='CDE', experiment_pattern='hyperparam'):
        """Initialize with the specific parameters from run_narrow.sh"""
        # Map dataset names from run_narrow.sh format to config format
        dataset_name_mapping = {
            'PBMC': 'pbmc',  # run_narrow.sh uses PBMC, but config uses pbmc
            'GoM': 'GoM',
            'LV': 'LV', 
            'Repressilator': 'Repressilator'
        }
        
        # Use the mapped name for the config
        config_dataset_name = dataset_name_mapping.get(dataset_name, dataset_name)
        
        super().__init__(
            dataset_name=config_dataset_name,
            forecast_method=forecast_method,
            predictor_type='dt_ridge_sinusoidal',
            sampling_type='dt_equals_one', 
            predictor_loss_weight=0,
            experiment_pattern=experiment_pattern
        )
        self.target_seeds = list(range(10))  # Seeds 0-9 as specified in run_narrow.sh
        self.original_dataset_name = dataset_name  # Keep the original name for directory search
        self.dataset_name_mapping = dataset_name_mapping  # Store the mapping for later use
        # Explicit snapMMD seeds as used in load_results_unified.py
        self.snapmmd_seeds = [1, 2, 3, 4, 5, 40, 41, 42, 43, 44]
        
    def filter_experiments_for_narrow_params(self, experiment_dirs):
        """
        Filter experiment directories to match exactly the parameters from run_narrow.sh.
        
        Returns:
            List of experiment directories that match our target parameters
        """
        matching_experiments = []
        
        for exp_dir in experiment_dirs:
            try:
                hyperparams = self.extract_hyperparameters_from_config(exp_dir)
                
                # Check if this experiment matches our target parameters
                if (hyperparams['predictor'] == 'dt_ridge_sinusoidal' and
                    hyperparams['sampling'] == 'dt_equals_one' and
                    hyperparams['predictor_loss_weight'] == 0 and
                    hyperparams['seed'] in self.target_seeds):
                    
                    matching_experiments.append({
                        'dir': exp_dir,
                        'seed': hyperparams['seed'],
                        'hyperparams': hyperparams
                    })
                    
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to extract hyperparameters from {exp_dir}: {e}")
                else:
                    print(f"Warning: Failed to extract hyperparameters from {exp_dir}: {e}")
                continue
        
        # Sort by seed for consistent ordering
        matching_experiments.sort(key=lambda x: x['seed'])
        
        return matching_experiments
    
    def load_snapmmd_forecast_for_seed(self, dataset_name, seed):
        """
        Load snapMMD forecast for a specific seed, matching the logic from load_results_unified.py
        
        Args:
            dataset_name: Name of the dataset (config format)
            seed: Random seed
            
        Returns:
            Dictionary with forecast data
        """
        try:
            if self.logger:
                self.logger.info(f"Loading snapMMD forecast for {dataset_name} with seed {seed}...")
            
            forecast_data = np.load(f"snapMMD_forecasts/{dataset_name}_forecast_{seed}.npz")
            
            # Extract forecast results - take only the second element (index 1:)
            # snapMMD forecast has shape (2, N, D), we want (1, N, D)
            forecast = forecast_data['forecast'][1:]  # Take second element only
            X_val_forecast = forecast_data['X_val']
            
            if self.logger:
                self.logger.info(f"Loaded snapMMD forecast for seed {seed}, shape: {forecast.shape}")
            
            return {
                'forecast': forecast,
                'X_val_forecast': X_val_forecast
            }
            
        except FileNotFoundError:
            if self.logger:
                self.logger.warning(f"snapMMD forecast file not found: snapMMD_forecasts/{dataset_name}_forecast_{seed}.npz")
            else:
                print(f"Warning: snapMMD forecast file not found: snapMMD_forecasts/{dataset_name}_forecast_{seed}.npz")
            return None
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading snapMMD forecast for seed {seed}: {e}")
            else:
                print(f"Error loading snapMMD forecast for seed {seed}: {e}")
            return None
    
    def compute_average_forecast_from_group(self, experiment_group, dataset_name):
        """
        Override parent method to handle dataset name mapping correctly.
        Compute average forecast from a group of experiments with the same hyperparameters but different seeds.
        
        Args:
            experiment_group: List of experiment dictionaries for the same hyperparameter set
            dataset_name: Name of the dataset (original format from run_narrow.sh)
            
        Returns:
            Dictionary with averaged forecast data and individual seed results
        """
        forecasts = []
        mmd_scores = []
        emd_scores = []
        valid_seeds = []
        
        # Use the mapped dataset name for config compatibility
        config_dataset_name = self.dataset_name_mapping.get(dataset_name, dataset_name)
        
        if self.forecast_method == 'snapMMD':
            # For snapMMD, load forecast files directly based on seeds
            for exp_data in experiment_group:
                try:
                    seed = exp_data['seed']
                    
                    # Load snapMMD forecast for this seed
                    snapmmd_forecast = self.load_snapmmd_forecast_for_seed(config_dataset_name, seed)
                    if snapmmd_forecast is None:
                        continue
                        
                    forecast_data = snapmmd_forecast['forecast']
                    forecasts.append(forecast_data)
                    valid_seeds.append(seed)
                    
                    # Calculate MMD and EMD for this seed
                    mmd_result = calculate_mmd_scores(config_dataset_name, seeds=[seed], forecast_method='snapMMD', logger=None)
                    if mmd_result:
                        mmd_scores.append(mmd_result['mean_mmd_squared'])
                    
                    if DATASET_CONFIGS[config_dataset_name]['calculate_emd']:
                        emd_result = calculate_emd_scores(config_dataset_name, seeds=[seed], forecast_method='snapMMD', logger=None)
                        if emd_result:
                            emd_scores.append(emd_result['mean_emd'])
                
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Failed to load snapMMD forecast for seed {exp_data['seed']}: {e}")
                    else:
                        print(f"Warning: Failed to load snapMMD forecast for seed {exp_data['seed']}: {e}")
                    continue
        
        elif self.forecast_method == 'CDE':
            # For CDE, load from experiment directories
            for exp_data in experiment_group:
                try:
                    # Set up a temporary loader for this experiment
                    exp_dir = exp_data['dir']
                    hyperparams = exp_data['hyperparams']
                    
                    temp_loader = UnifiedResultsLoader(
                        config_dataset_name,  # Use mapped name here
                        forecast_method=self.forecast_method,
                        predictor_type=hyperparams['predictor'],
                        sampling_type=hyperparams['sampling'],
                        predictor_loss_weight=hyperparams['predictor_loss_weight'],
                        experiment_pattern=self.experiment_pattern
                    )
                    
                    # Set the experiment directory for CDE forecasting
                    temp_loader.experiment_dir = exp_dir
                    
                    # Load data and forecast for this experiment
                    results = temp_loader.load_data_and_forecast()
                    forecast_data = results['forecast_data']['forecast']
                    
                    forecasts.append(forecast_data)
                    valid_seeds.append(exp_data['seed'])
                    
                    # Calculate MMD and EMD for this individual experiment
                    cde_forecast = results['forecast_data']['forecast']
                    
                    mmd_result = calculate_mmd_scores(config_dataset_name, forecast_method=self.forecast_method, 
                                                     cde_forecast_data=cde_forecast, logger=None)
                    if mmd_result:
                        mmd_scores.append(mmd_result['mean_mmd_squared'])
                    
                    if DATASET_CONFIGS[config_dataset_name]['calculate_emd']:
                        emd_result = calculate_emd_scores(config_dataset_name, forecast_method=self.forecast_method, 
                                                         cde_forecast_data=cde_forecast, logger=None)
                        if emd_result:
                            emd_scores.append(emd_result['mean_emd'])
                    
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Failed to load forecast from {exp_data['dir']} (seed {exp_data['seed']}): {e}")
                    else:
                        print(f"Warning: Failed to load forecast from {exp_data['dir']} (seed {exp_data['seed']}): {e}")
                    continue
        
        else:
            raise ValueError(f"Unsupported forecast method: {self.forecast_method}")
        
        if not forecasts:
            raise ValueError(f"No valid forecasts found in experiment group")
        
        # Convert to numpy arrays and compute average
        forecasts_array = np.array(forecasts)  # Shape: (n_seeds, n_timesteps, n_particles, n_features)
        average_forecast = np.mean(forecasts_array, axis=0)  # Average over seeds
        
        # Prepare results structure similar to what load_data_and_forecast returns
        # We'll use the first experiment's training data since it should be the same across seeds
        first_exp_dir = experiment_group[0]['dir']
        first_hyperparams = experiment_group[0]['hyperparams']
        
        if self.forecast_method == 'snapMMD':
            # For snapMMD, we need to load training data but use snapMMD forecast format
            temp_loader = UnifiedResultsLoader(
                config_dataset_name,
                seed=valid_seeds[0],  # Use first valid seed for snapMMD
                forecast_method='snapMMD',  # Use snapMMD to load the forecast correctly
                predictor_type=first_hyperparams['predictor'],
                sampling_type=first_hyperparams['sampling'],
                predictor_loss_weight=first_hyperparams['predictor_loss_weight'],
                experiment_pattern=self.experiment_pattern
            )
            # Load training data and first forecast for structure reference
            first_results = temp_loader.load_data_and_forecast()
            
        elif self.forecast_method == 'CDE':
            # For CDE, use the experiment directory
            temp_loader = UnifiedResultsLoader(
                config_dataset_name,  # Use mapped name here too
                forecast_method=self.forecast_method,
                predictor_type=first_hyperparams['predictor'],
                sampling_type=first_hyperparams['sampling'],
                predictor_loss_weight=first_hyperparams['predictor_loss_weight'],
                experiment_pattern=self.experiment_pattern
            )
            temp_loader.experiment_dir = first_exp_dir
            first_results = temp_loader.load_data_and_forecast()
        
        # Create averaged results
        averaged_results = {
            'training_data': first_results['training_data'],  # Same across all seeds
            'forecast_data': {
                'forecast': average_forecast,
                'X_val_forecast': first_results['forecast_data']['X_val_forecast']  # Same across all seeds
            },
            'metadata': {
                'task_name': config_dataset_name,  # Use mapped name
                'config': first_results['metadata']['config'],
                'forecast_method': self.forecast_method,
                'averaged_over_seeds': valid_seeds,
                'n_seeds': len(valid_seeds)
            },
            'individual_results': {
                'mmd_scores': mmd_scores,
                'emd_scores': emd_scores,
                'seeds': valid_seeds
            }
        }
        
        return averaged_results
    
    def analyze_dataset_across_seeds(self, dataset_name, base_outputs_dir, skip_plots=False, skip_metrics=False, output_folder="figures"):
        """
        Analyze a single dataset across all seeds with the narrow parameters.
        
        Args:
            dataset_name: Name of the dataset to analyze
            base_outputs_dir: Base directory containing experiment outputs
            skip_plots: Whether to skip plot generation
            skip_metrics: Whether to skip metrics calculation
            output_folder: Where to save output files
            
        Returns:
            Dictionary with analysis results
        """
        if self.logger:
            self.logger.info(f"Starting narrow analysis for dataset: {dataset_name}")
        else:
            print(f"Starting narrow analysis for dataset: {dataset_name}")
        
        # Build the list of runs to process
        if self.forecast_method == 'snapMMD':
            # For snapMMD, do not inspect configs; use explicit seed list
            matching_experiments = [{ 'seed': s } for s in self.snapmmd_seeds]
            found_seeds = self.snapmmd_seeds
            if self.logger:
                self.logger.info(f"Using snapMMD seeds: {found_seeds}")
            else:
                print(f"Using snapMMD seeds: {found_seeds}")
        else:
            # For CDE, find experiment directories and filter by exact hyperparameters
            try:
                search_dataset_name = self.original_dataset_name if hasattr(self, 'original_dataset_name') else dataset_name
                all_experiment_dirs = self.find_all_experiments_with_hash(search_dataset_name, base_outputs_dir, self.experiment_pattern)
                if self.logger:
                    self.logger.info(f"Found {len(all_experiment_dirs)} total experiments for {dataset_name}")
                else:
                    print(f"Found {len(all_experiment_dirs)} total experiments for {dataset_name}")
            except Exception as e:
                error_msg = f"Failed to find experiments for {dataset_name}: {e}"
                if self.logger:
                    self.logger.error(error_msg)
                else:
                    print(f"Error: {error_msg}")
                return None
            
            matching_experiments = self.filter_experiments_for_narrow_params(all_experiment_dirs)
            
            if not matching_experiments:
                error_msg = f"No experiments found for {dataset_name} with target parameters (dt_ridge_sinusoidal + dt_equals_one + predictor_loss_weight=0)"
                if self.logger:
                    self.logger.error(error_msg)
                else:
                    print(f"Error: {error_msg}")
                return None
            
            found_seeds = [exp['seed'] for exp in matching_experiments]
            missing_seeds = set(self.target_seeds) - set(found_seeds)
            
            if self.logger:
                self.logger.info(f"Found {len(matching_experiments)} matching experiments with seeds: {found_seeds}")
                if missing_seeds:
                    self.logger.warning(f"Missing experiments for seeds: {sorted(missing_seeds)}")
            else:
                print(f"Found {len(matching_experiments)} matching experiments with seeds: {found_seeds}")
                if missing_seeds:
                    print(f"Warning: Missing experiments for seeds: {sorted(missing_seeds)}")
        
        # Set up PCA if needed (for PBMC)
        self.setup_pca_if_needed()
        
        # Compute average forecast and individual metrics
        try:
            results = self.compute_average_forecast_from_group(matching_experiments, dataset_name)
            
            if self.logger:
                self.logger.info(f"Successfully computed averaged results across {len(found_seeds)} seeds")
            else:
                print(f"Successfully computed averaged results across {len(found_seeds)} seeds")
                
        except Exception as e:
            error_msg = f"Failed to compute average forecast for {dataset_name}: {e}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(f"Error: {error_msg}")
            return None
        
        # Generate plots
        if not skip_plots:
            try:
                if self.logger:
                    self.logger.info("Generating plots...")
                else:
                    print("Generating plots...")
                
                self.plot_main_results(results, output_folder)
                self.plot_trajectories(results, output_folder)
                self.plot_interactive_3d(results, output_folder)
                self.plot_multi_angle_views(results, output_folder)
                self.plot_individual_final_timepoints(results, output_folder)
                
                if self.logger:
                    self.logger.info("All plots generated successfully")
                else:
                    print("All plots generated successfully")
                    
            except Exception as e:
                error_msg = f"Error generating plots for {dataset_name}: {e}"
                if self.logger:
                    self.logger.error(error_msg)
                else:
                    print(f"Error: {error_msg}")
        
        # Calculate and log metrics
        if not skip_metrics:
            try:
                if self.logger:
                    self.logger.info("Computing grouped metrics...")
                else:
                    print("Computing grouped metrics...")
                
                # Extract individual results
                individual_results = results['individual_results']
                mmd_scores = individual_results['mmd_scores']
                emd_scores = individual_results['emd_scores']
                valid_seeds = individual_results['seeds']
                
                # Log individual seed results
                if self.logger:
                    self.logger.info(f"Individual MMD^2 scores by seed:")
                    for seed, mmd_score in zip(valid_seeds, mmd_scores):
                        mmd = np.sqrt(mmd_score)
                        self.logger.info(f"  Seed {seed}: MMD = {mmd:.6f}, MMD^2 = {mmd_score:.6f}")
                else:
                    print("Individual MMD^2 scores by seed:")
                    for seed, mmd_score in zip(valid_seeds, mmd_scores):
                        mmd = np.sqrt(mmd_score)
                        print(f"  Seed {seed}: MMD = {mmd:.6f}, MMD^2 = {mmd_score:.6f}")
                        
                if emd_scores:
                    if self.logger:
                        self.logger.info(f"Individual EMD scores by seed:")
                        for seed, emd_score in zip(valid_seeds, emd_scores):
                            self.logger.info(f"  Seed {seed}: EMD = {emd_score:.6f}")
                    else:
                        print("Individual EMD scores by seed:")
                        for seed, emd_score in zip(valid_seeds, emd_scores):
                            print(f"  Seed {seed}: EMD = {emd_score:.6f}")
                
                # Compute and log statistics
                if mmd_scores:
                    mmd_squared_array = np.array(mmd_scores)
                    mmd_array = np.sqrt(mmd_squared_array)
                    
                    mean_mmd = np.mean(mmd_array)
                    std_mmd = np.std(mmd_array)
                    mean_mmd_squared = np.mean(mmd_squared_array)
                    std_mmd_squared = np.std(mmd_squared_array)
                    
                    if self.logger:
                        self.logger.info(f"MMD Statistics (n={len(mmd_scores)}):")
                        self.logger.info(f"  MMD: {mean_mmd:.6f} ± {std_mmd:.6f}")
                        self.logger.info(f"  MMD^2: {mean_mmd_squared:.6f} ± {std_mmd_squared:.6f}")
                    
                    print(f"MMD Statistics (n={len(mmd_scores)}):")
                    print(f"  MMD: {mean_mmd:.6f} ± {std_mmd:.6f}")
                    print(f"  MMD^2: {mean_mmd_squared:.6f} ± {std_mmd_squared:.6f}")
                
                if emd_scores:
                    emd_array = np.array(emd_scores)
                    mean_emd = np.mean(emd_array)
                    std_emd = np.std(emd_array)
                    
                    if self.logger:
                        self.logger.info(f"EMD Statistics (n={len(emd_scores)}):")
                        self.logger.info(f"  EMD: {mean_emd:.6f} ± {std_emd:.6f}")
                    
                    print(f"EMD Statistics (n={len(emd_scores)}):")
                    print(f"  EMD: {mean_emd:.6f} ± {std_emd:.6f}")
                
                # Determine missing seeds based on method
                if self.forecast_method == 'snapMMD':
                    target = set(self.snapmmd_seeds)
                else:
                    target = set(self.target_seeds)
                missing_seeds = sorted(list(target - set(valid_seeds)))

                # Store metrics in results
                results['summary_metrics'] = {
                    'mmd_stats': {
                        'mean': mean_mmd,
                        'std': std_mmd,
                        'mean_squared': mean_mmd_squared,
                        'std_squared': std_mmd_squared,
                        'count': len(mmd_scores)
                    } if mmd_scores else None,
                    'emd_stats': {
                        'mean': mean_emd,
                        'std': std_emd,
                        'count': len(emd_scores)
                    } if emd_scores else None,
                    'valid_seeds': valid_seeds,
                    'missing_seeds': sorted(missing_seeds)
                }
                
            except Exception as e:
                error_msg = f"Error computing metrics for {dataset_name}: {e}"
                if self.logger:
                    self.logger.error(error_msg)
                else:
                    print(f"Error: {error_msg}")
        
        return results

def main():
    """Main function for narrow analysis of run_narrow.sh results."""
    parser = argparse.ArgumentParser(
        description='Analyze results from run_narrow.sh: compute mean ± std across 10 seeds for specific hyperparameters. Supports both snapMMD and CDE methods.'
    )
    parser.add_argument('--datasets', nargs='+', 
                       choices=['GoM', 'PBMC', 'LV', 'Repressilator'],
                       default=['GoM', 'PBMC', 'LV', 'Repressilator'],
                       help='Dataset names to process (default: all 4 datasets from run_narrow.sh)')
    parser.add_argument('--forecast-method', type=str, default='CDE', 
                       choices=['snapMMD', 'CDE'],
                       help='Forecasting method: snapMMD (load pre-computed forecasts) or CDE (generate from models) (default: CDE)')
    parser.add_argument('--output-folder', type=str, default='figures_narrow',
                       help='Output folder for figures (default: figures_narrow)')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip plotting, only calculate metrics')
    parser.add_argument('--skip-metrics', action='store_true', 
                       help='Skip metrics calculation, only plot')
    parser.add_argument('--experiment-pattern', type=str, default='hyperparam',
                       choices=['hyperparam', 'unified'],
                       help='Experiment directory pattern (default: hyperparam)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("NARROW ANALYSIS OF run_narrow.sh RESULTS")
    print("="*80)
    print(f"Target parameters:")
    print(f"  - Predictor: dt_ridge_sinusoidal")
    print(f"  - Sampling: dt_equals_one")
    print(f"  - Predictor loss weight: 0")
    print(f"  - Seeds: 0-9 (10 seeds)")
    print(f"Processing datasets: {args.datasets}")
    print(f"Using forecasting method: {args.forecast_method}")
    print(f"Output folder: {args.output_folder}")
    print("="*80)

    original_cwd = "/orcd/archive/abugoot/001/Projects/paolo/CoupledDistributionEmbeddings/"
    base_outputs_dir = os.path.join(original_cwd, 'outputs')
    
    # Create output directory
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Store results for summary
    all_results = {}
    
    # Process each dataset
    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"PROCESSING DATASET: {dataset_name}")
        print(f"{'='*60}")
        
        # Create analyzer for this dataset
        analyzer = NarrowAnalysisLoader(
            dataset_name=dataset_name,
            forecast_method=args.forecast_method,
            experiment_pattern=args.experiment_pattern
        )
        
        # Set up logging
        log_filepath = analyzer.setup_logging(args.output_folder)
        print(f"Logging to: {log_filepath}")
        
        # Analyze this dataset
        try:
            results = analyzer.analyze_dataset_across_seeds(
                dataset_name=dataset_name,
                base_outputs_dir=base_outputs_dir,
                skip_plots=args.skip_plots,
                skip_metrics=args.skip_metrics,
                output_folder=args.output_folder
            )
            
            if results is not None:
                all_results[dataset_name] = results
                
                # Get output paths
                folder_name = f"{dataset_name}_dt_ridge_sinusoidal_dt_equals_one_0"
                figures_path = os.path.join(args.output_folder, folder_name)
                
                valid_seeds = results['individual_results']['seeds']
                completion_message = f"✓ Analysis complete for {dataset_name} ({len(valid_seeds)} seeds)"
                
                analyzer.logger.info(completion_message)
                analyzer.logger.info(f"Figures saved to: {figures_path}/")
                analyzer.logger.info(f"Analysis log saved to: {log_filepath}")
                
                print(completion_message)
                print(f"  Figures: {figures_path}/")
                print(f"  Log: {log_filepath}")
                
            else:
                print(f"✗ Failed to analyze {dataset_name}")
                
        except Exception as e:
            print(f"✗ Error processing dataset {dataset_name}: {e}")
            continue
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    if all_results:
        print(f"Successfully processed {len(all_results)} out of {len(args.datasets)} datasets:")
        
        for dataset_name, results in all_results.items():
            if 'summary_metrics' in results:
                metrics = results['summary_metrics']
                valid_seeds = metrics['valid_seeds']
                missing_seeds = metrics['missing_seeds']
                
                print(f"\n{dataset_name}:")
                print(f"  Seeds processed: {len(valid_seeds)}/10 {valid_seeds}")
                if missing_seeds:
                    print(f"  Missing seeds: {missing_seeds}")
                
                if metrics['mmd_stats']:
                    mmd_stats = metrics['mmd_stats']
                    print(f"  MMD: {mmd_stats['mean']:.6f} ± {mmd_stats['std']:.6f}")
                    print(f"  MMD²: {mmd_stats['mean_squared']:.6f} ± {mmd_stats['std_squared']:.6f}")
                
                if metrics['emd_stats']:
                    emd_stats = metrics['emd_stats']
                    print(f"  EMD: {emd_stats['mean']:.6f} ± {emd_stats['std']:.6f}")
        
        print(f"\nAll results saved to: {args.output_folder}/")
        
    else:
        print("No datasets were successfully processed.")
    
    print(f"{'='*80}")
    print("NARROW ANALYSIS COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
