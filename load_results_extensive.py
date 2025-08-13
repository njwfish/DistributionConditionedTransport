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
from utils.snapMMD import MMDLoss, RBF
from scipy.optimize import linprog
from sklearn.decomposition import PCA
import plotly.graph_objects as go

# Add imports needed for CDE mode
from utils.experiment_utils import load_best_model, get_experiment_info
from latent_mapping_training import load_latent_mapping_model

# Dataset configurations
DATASET_CONFIGS = {
    'LV': {
        'name': 'LV',
        'data_path': 'data/classic/LV_data.npz',
        'experiment_type': 'classic',
        'dimensionality': 2,
        'axes_labels': ['Prey', 'Predator'],
        'title': 'Lotka-Volterra',
        'calculate_emd': True,
        'special_plots': []
    },
    'Repressilator': {
        'name': 'Repressilator', 
        'data_path': 'data/classic/Repressilator_data.npz',
        'experiment_type': 'classic',
        'dimensionality': 3,
        'axes_labels': ['Gene 1', 'Gene 2', 'Gene 3'],
        'title': 'Repressilator',
        'calculate_emd': True,
        'special_plots': ['interactive_html', 'multi_angle']
    },
    'GoM': {
        'name': 'GoM',
        'data_path': 'data/realdata/GoM_data.npz', 
        'experiment_type': 'realdata',
        'dimensionality': 2,
        'axes_labels': ['X1', 'X2'],
        'title': 'GoM',
        'calculate_emd': True,
        'special_plots': []
    },
    'pbmc': {
        'name': 'pbmc',
        'data_path': 'data/realdata/processed_pbmc_data_sub500_every_2_until20.npz',
        'experiment_type': 'realdata',
        'dimensionality': 30,  # Original dimensionality
        'plot_dimensionality': 3,  # After PCA
        'axes_labels': ['PC1', 'PC2', 'PC3'],
        'title': 'PBMC',
        'calculate_emd': False,  # doesn't converge according to the snapMMD paper
        'special_plots': ['interactive_html', 'multi_angle', 'individual_final'],
        'requires_pca': True
    }
}

class UnifiedResultsLoader:
    def __init__(self, dataset_name, seed=42, forecast_method='snapMMD', 
                 predictor_type=None, sampling_type=None, predictor_loss_weight=None, experiment_pattern='hyperparam'):
        """
        Initialize the unified results loader.
        
        Args:
            dataset_name: Dataset name
            seed: Random seed for snapMMD method
            forecast_method: 'snapMMD' or 'CDE'
            predictor_type: Predictor type for hyperparameter experiments (e.g., 'dt_mlp_sinusoidal')
            sampling_type: Sampling type for hyperparameter experiments (e.g., 'bidirectional')
            predictor_loss_weight: Predictor loss weight for hyperparameter experiments (e.g., 0.01)
            experiment_pattern: Pattern type for finding experiments ('hyperparam' or 'unified')
        """
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Dataset {dataset_name} not supported. Choose from: {list(DATASET_CONFIGS.keys())}")
        
        if forecast_method not in ['snapMMD', 'CDE']:
            raise ValueError(f"forecast_method must be 'snapMMD' or 'CDE', got '{forecast_method}'")
        
        self.config = DATASET_CONFIGS[dataset_name]
        self.dataset_name = dataset_name
        self.forecast_method = forecast_method
        self.predictor_type = predictor_type
        self.sampling_type = sampling_type
        self.predictor_loss_weight = predictor_loss_weight
        self.experiment_pattern = experiment_pattern
        
        # Seed is only relevant for snapMMD method
        if forecast_method == 'snapMMD':
            self.seed = seed
        else:
            self.seed = None
            
        self.pca = None
        self.logger = None
    
    def setup_logging(self, output_folder="figures"):
        """Set up logging to file in the nested directory structure."""
        # Create nested directory structure: figures/<dataset>_<predictor>_<sampling>_<predictor_loss_weight>/
        if self.predictor_type and self.sampling_type and self.predictor_loss_weight is not None:
            folder_name = f"{self.dataset_name}_{self.predictor_type}_{self.sampling_type}_{self.predictor_loss_weight}"
        elif self.predictor_type and self.sampling_type:
            folder_name = f"{self.dataset_name}_{self.predictor_type}_{self.sampling_type}"
        else:
            folder_name = f"{self.dataset_name}_{self.forecast_method}"
        
        nested_output_folder = os.path.join(output_folder, folder_name)
        os.makedirs(nested_output_folder, exist_ok=True)
        
        # Create log filename
        if self.predictor_type and self.sampling_type and self.predictor_loss_weight is not None:
            log_filename = f"{self.dataset_name}_{self.predictor_type}_{self.sampling_type}_{self.predictor_loss_weight}_analysis_{self.forecast_method}.log"
        elif self.predictor_type and self.sampling_type:
            log_filename = f"{self.dataset_name}_{self.predictor_type}_{self.sampling_type}_analysis_{self.forecast_method}.log"
        elif self.seed is not None:
            log_filename = f"{self.dataset_name}_analysis_seed_{self.seed}_{self.forecast_method}.log"
        else:
            log_filename = f"{self.dataset_name}_analysis_{self.forecast_method}.log"
        
        log_filepath = os.path.join(nested_output_folder, log_filename)
        
        # Set up logger
        self.logger = logging.getLogger(f'{self.dataset_name}_{self.forecast_method}')
        self.logger.setLevel(logging.INFO)
        
        # Remove any existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create file handler
        file_handler = logging.FileHandler(log_filepath, mode='w')
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                    datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
        return log_filepath
        
    def find_all_experiments_with_hash(self, dataset_name, base_dir, pattern_type='hyperparam'):
        """
        Find ALL experiment directories that match the pattern for different datasets
        
        Args:
            dataset_name: One of ["LV", "Repressilator", "GoM", "PBMC"]
            base_dir: The outputs directory containing experiment directories
            pattern_type: Type of pattern to match ('hyperparam' or 'unified')
            
        Returns:
            List of full paths to all matching experiment directories
        """
        # Handle special case for pbmc -> PBMC mapping (experiment directories use uppercase)
        search_name = dataset_name
        if dataset_name == 'pbmc':
            search_name = 'PBMC'
        
        # Choose pattern based on pattern_type
        if pattern_type == 'hyperparam':
            pattern = f"snapMMD_{search_name}_unified_hyperparam_exp_"
            expected_bare = f"snapMMD_{search_name}_unified_hyperparam_exp"
        elif pattern_type == 'unified':
            pattern = f"snapMMD_{search_name}_unified_exp_"
            expected_bare = f"snapMMD_{search_name}_unified_exp"
        else:
            raise ValueError(f"Unknown pattern_type: {pattern_type}. Must be 'hyperparam' or 'unified'")
        
        if not os.path.exists(base_dir):
            raise ValueError(f"Base directory {base_dir} does not exist")
        
        matching_dirs = []
        for item in os.listdir(base_dir):
            full_path = os.path.join(base_dir, item)
            if os.path.isdir(full_path) and item.startswith(pattern):
                # We want the ones with the hash, not the bare name
                if item != expected_bare:
                    matching_dirs.append(full_path)
        
        if len(matching_dirs) == 0:
            raise ValueError(f"No experiment directories found matching pattern {pattern}<hash>")
        
        if self.logger:
            self.logger.info(f"Found {len(matching_dirs)} experiment directories for {dataset_name}")
        else:
            print(f"Found {len(matching_dirs)} experiment directories for {dataset_name}")
        
        return matching_dirs

    def find_experiment_with_hash(self, dataset_name, base_dir, pattern_type='hyperparam'):
        """
        Find the experiment directory that matches the pattern for different datasets
        (kept for backwards compatibility, but now just returns the first one)
        
        Args:
            dataset_name: One of ["LV", "Repressilator", "GoM", "PBMC"]
            base_dir: The outputs directory containing experiment directories
            pattern_type: Type of pattern to match ('hyperparam' or 'unified')
            
        Returns:
            The full path to the matching experiment directory
        """
        all_dirs = self.find_all_experiments_with_hash(dataset_name, base_dir, pattern_type)
        return all_dirs[0]

    def extract_hyperparameters_from_config(self, experiment_dir):
        """
        Extract predictor, sampling hyperparameters, and predictor_loss_weights from config.yaml
        
        Args:
            experiment_dir: Path to experiment directory containing config.yaml
            
        Returns:
            Dictionary with 'predictor', 'sampling', 'predictor_loss_weights', and 'seed' keys
            
        Raises:
            ValueError: If experiment.use_predicted_latent is True (experiment should be skipped)
        """
        config_path = os.path.join(experiment_dir, 'config.yaml')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.yaml not found in {experiment_dir}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check if experiment.use_predicted_latent is True and skip if so
        if 'experiment' in config and isinstance(config['experiment'], dict):
            use_predicted_latent = config['experiment'].get('use_predicted_latent', False)
            if use_predicted_latent:
                raise ValueError(f"Skipping experiment with use_predicted_latent=True: {experiment_dir}")
        
        # Extract predictor type - fail if not found
        if 'predictor' not in config:
            raise KeyError(f"'predictor' key not found in config.yaml at {experiment_dir}")
        
        predictor_config = config['predictor']
        if not isinstance(predictor_config, dict) or '_target_' not in predictor_config:
            raise ValueError(f"Invalid predictor configuration in {experiment_dir}: expected dict with '_target_' key")
        
        target = predictor_config['_target_']
        if 'DTConditionedMLPPredictor' in target:
            # New DTConditionedMLPPredictor structure
            if 'conditioning_mode' not in predictor_config:
                raise KeyError(f"'conditioning_mode' not found in predictor config at {experiment_dir}")
            conditioning_mode = predictor_config['conditioning_mode']
            predictor_type = f"dt_mlp_{conditioning_mode}"
        elif 'DTConditionedRidgePredictor' in target:
            # New DTConditionedRidgePredictor structure
            if 'conditioning_mode' not in predictor_config:
                raise KeyError(f"'conditioning_mode' not found in predictor config at {experiment_dir}")
            conditioning_mode = predictor_config['conditioning_mode']
            predictor_type = f"dt_ridge_{conditioning_mode}"
        elif 'DTMLPPredictor' in target:
            # Legacy DTMLPPredictor structure (fallback)
            if 'conditioning_method' not in predictor_config:
                raise KeyError(f"'conditioning_method' not found in predictor config at {experiment_dir}")
            conditioning_method = predictor_config['conditioning_method']
            predictor_type = f"dt_mlp_{conditioning_method}"
        elif 'DTRidgePredictor' in target:
            # Legacy DTRidgePredictor structure (fallback)
            if 'conditioning_method' not in predictor_config:
                raise KeyError(f"'conditioning_method' not found in predictor config at {experiment_dir}")
            conditioning_method = predictor_config['conditioning_method']
            predictor_type = f"dt_ridge_{conditioning_method}"
        elif 'MLPPredictor' in target:
            # Regular MLPPredictor (non-dt-conditioned)
            predictor_type = 'mlp'
        elif 'RidgePredictor' in target:
            # Regular RidgePredictor (non-dt-conditioned)
            predictor_type = 'ridge'
        else:
            raise ValueError(f"Unknown predictor target '{target}' in {experiment_dir}")
        
        # Extract sampling type - fail if not found
        if 'sampling' not in config:
            raise KeyError(f"'sampling' key not found in config.yaml at {experiment_dir}")
        
        sampling_config = config['sampling']
        if not isinstance(sampling_config, dict):
            raise ValueError(f"Invalid sampling configuration in {experiment_dir}: expected dict")
        
        # Sampling configs use 'mode' field instead of '_target_'
        if 'mode' not in sampling_config:
            raise KeyError(f"'mode' key not found in sampling config at {experiment_dir}")
        
        mode = sampling_config['mode']
        if mode == 'bidirectional':
            sampling_type = 'bidirectional'
        elif mode == 'unidirectional':
            sampling_type = 'unidirectional'
        elif mode == 'exponential':
            sampling_type = 'exponential'
        elif mode == 'dt_equals_one':
            sampling_type = 'dt_equals_one'
        elif mode is None or mode == 'none':
            sampling_type = 'none'
        else:
            raise ValueError(f"Unknown sampling mode '{mode}' in {experiment_dir}")
        
        # Extract predictor_loss_weights from experiment config
        if 'experiment' not in config:
            raise KeyError(f"'experiment' key not found in config.yaml at {experiment_dir}")
        
        experiment_config = config['experiment']
        if not isinstance(experiment_config, dict):
            raise ValueError(f"Invalid experiment configuration in {experiment_dir}: expected dict")
        
        if 'predictor_loss_weight' not in experiment_config:
            raise KeyError(f"'predictor_loss_weight' not found in experiment config at {experiment_dir}")
        
        predictor_loss_weight = experiment_config['predictor_loss_weight']
        
        # Extract seed
        if 'seed' not in config:
            raise KeyError(f"'seed' key not found in config.yaml at {experiment_dir}")
        
        seed = config['seed']
        
        return {
            'predictor': predictor_type,
            'sampling': sampling_type,
            'predictor_loss_weight': predictor_loss_weight,
            'seed': seed
        }

    def group_experiments_by_hyperparameters(self, experiment_dirs):
        """
        Group experiments by hyperparameters (excluding seed).
        
        Args:
            experiment_dirs: List of experiment directory paths
            
        Returns:
            Dictionary mapping hyperparameter tuples to lists of experiment directories
        """
        groups = {}
        
        for exp_dir in experiment_dirs:
            try:
                hyperparams = self.extract_hyperparameters_from_config(exp_dir)
                
                # Create key excluding seed
                key = (
                    hyperparams['predictor'], 
                    hyperparams['sampling'], 
                    hyperparams['predictor_loss_weight']
                )
                
                if key not in groups:
                    groups[key] = []
                
                groups[key].append({
                    'dir': exp_dir,
                    'seed': hyperparams['seed'],
                    'hyperparams': hyperparams
                })
                
            except ValueError as e:
                # Handle experiments that should be skipped (use_predicted_latent=True)
                if "use_predicted_latent=True" in str(e):
                    if self.logger:
                        self.logger.info(f"Skipping experiment with use_predicted_latent=True: {exp_dir}")
                    else:
                        print(f"Skipping experiment with use_predicted_latent=True: {exp_dir}")
                else:
                    if self.logger:
                        self.logger.warning(f"Failed to extract hyperparameters from {exp_dir}: {e}")
                    else:
                        print(f"Warning: Failed to extract hyperparameters from {exp_dir}: {e}")
                continue
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to extract hyperparameters from {exp_dir}: {e}")
                else:
                    print(f"Warning: Failed to extract hyperparameters from {exp_dir}: {e}")
                continue
        
        # Sort experiments within each group by seed for consistent ordering
        for key in groups:
            groups[key].sort(key=lambda x: x['seed'])
        
        return groups

    def compute_average_forecast_from_group(self, experiment_group, dataset_name):
        """
        Compute average forecast from a group of experiments with the same hyperparameters but different seeds.
        
        Args:
            experiment_group: List of experiment dictionaries for the same hyperparameter set
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with averaged forecast data and individual seed results
        """
        forecasts = []
        mmd_scores = []
        emd_scores = []
        valid_seeds = []
        
        for exp_data in experiment_group:
            try:
                # Set up a temporary loader for this experiment
                exp_dir = exp_data['dir']
                hyperparams = exp_data['hyperparams']
                
                temp_loader = UnifiedResultsLoader(
                    dataset_name,
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
                if self.forecast_method == 'CDE':
                    cde_forecast = results['forecast_data']['forecast']
                    
                    mmd_result = calculate_mmd_scores(dataset_name, forecast_method=self.forecast_method, 
                                                     cde_forecast_data=cde_forecast, logger=None)
                    if mmd_result:
                        mmd_scores.append(mmd_result['mean_mmd_squared'])
                    
                    if DATASET_CONFIGS[dataset_name]['calculate_emd']:
                        emd_result = calculate_emd_scores(dataset_name, forecast_method=self.forecast_method, 
                                                         cde_forecast_data=cde_forecast, logger=None)
                        if emd_result:
                            emd_scores.append(emd_result['mean_emd'])
                
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to load forecast from {exp_data['dir']} (seed {exp_data['seed']}): {e}")
                else:
                    print(f"Warning: Failed to load forecast from {exp_data['dir']} (seed {exp_data['seed']}): {e}")
                continue
        
        if not forecasts:
            raise ValueError(f"No valid forecasts found in experiment group")
        
        # Convert to numpy arrays and compute average
        forecasts_array = np.array(forecasts)  # Shape: (n_seeds, n_timesteps, n_particles, n_features)
        average_forecast = np.mean(forecasts_array, axis=0)  # Average over seeds
        
        # Prepare results structure similar to what load_data_and_forecast returns
        # We'll use the first experiment's training data since it should be the same across seeds
        first_exp_dir = experiment_group[0]['dir']
        first_hyperparams = experiment_group[0]['hyperparams']
        
        temp_loader = UnifiedResultsLoader(
            dataset_name,
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
                'task_name': dataset_name,
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

    def load_experiment_by_dataset(self, dataset_name, base_dir):
        """
        Load experiment configuration and directory for a given dataset.
        
        Args:
            dataset_name: One of ["LV", "Repressilator", "GoM", "pbmc"] (case-sensitive)
            base_dir: The outputs directory containing experiment directories
            
        Returns:
            Dictionary containing experiment info (config, dir, etc.)
        """
        valid_datasets = ["LV", "Repressilator", "GoM", "pbmc"]
        if dataset_name not in valid_datasets:
            raise ValueError(f"Dataset name must be one of {valid_datasets}, got {dataset_name}")
        
        # Map pbmc to PBMC for directory search (directories use uppercase)
        search_name = "PBMC" if dataset_name == "pbmc" else dataset_name
        
        experiment_dir = self.find_experiment_with_hash(search_name, base_dir, self.experiment_pattern)
        experiment_info = get_experiment_info(experiment_dir, load_checkpoints=False)
        
        return experiment_info

    def load_model_cde(self, cfg, path, device):
        """Load encoder and generator models for CDE forecasting."""
        enc = hydra.utils.instantiate(cfg['encoder'])
        gen = hydra.utils.instantiate(cfg['generator'])
        
        # Load predictor if it exists in config and attach to encoder
        if 'predictor' in cfg:
            predictor = hydra.utils.instantiate(cfg['predictor'])
            # Set the latent activation to match the encoder (as done in main.py)
            predictor.latent_act = enc.latent_act
            enc.predictor = predictor
        
        state = load_best_model(path)
        
        # Load encoder state
        enc.load_state_dict(state['encoder_state_dict'])
        
        gen.load_state_dict(state['generator_state_dict'])

        enc.eval()
        gen.eval()
        enc.to(device)
        gen.to(device)
        
        return enc, gen

    def generate_cde_forecast(self, training_data):
        """Generate forecast using CDE method."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # TODO: what exactly is experiment_dir supposed to be here? Make sure it is well-behaved.
        # Load experiment configuration
        if hasattr(self, 'experiment_dir') and self.experiment_dir:
            # Use specific experiment directory if set (for hyperparameter analysis)
            experiment_info = get_experiment_info(self.experiment_dir, load_checkpoints=False)
            if self.logger:
                self.logger.info(f"Loading CDE experiment from: {self.experiment_dir}")
            else:
                print(f"Loading CDE experiment from: {self.experiment_dir}")
        else:
            # Use original logic for backwards compatibility
            original_cwd = "/orcd/archive/abugoot/001/Projects/paolo/CoupledDistributionEmbeddings/"
            experiment_info = self.load_experiment_by_dataset(self.dataset_name, os.path.join(original_cwd, 'outputs'))
            if self.logger:
                self.logger.info(f"Loading CDE experiment from: {experiment_info['dir']}")
            else:
                print(f"Loading CDE experiment from: {experiment_info['dir']}")
        
        # Load encoder, generator, and optionally latent mapping model
        enc, gen = self.load_model_cde(experiment_info['config'], experiment_info['dir'], device)
        
        # Prepare data (use last two time points)
        Xs_training = training_data['Xs']
        samples_s = torch.tensor(Xs_training[-1]).unsqueeze(0).to(device).float()  # Last training point (source)
        samples_t = torch.tensor(training_data['X_val_true']).unsqueeze(0).to(device).float()  # Ground truth target
        
        # Generate encodings
        with torch.no_grad():
            enc_s = enc(samples_s)

            # Check if predictor requires dt conditioning (same logic as in loss/predictor.py)
            if hasattr(enc.predictor, 'requires_dt') and enc.predictor.requires_dt:
                # dt-conditioned predictor
                dt = torch.ones(enc_s.shape[0], device=device)
                enc_t = enc.predictor(enc_s, dt)
                if self.logger:
                    self.logger.info("Using dt-conditioned predictor")
            else:
                # non-dt-conditioned predictor
                enc_t = enc.predictor(enc_s)
                if self.logger:
                    self.logger.info("Using non-dt-conditioned predictor")

        # Reshape samples_s for forecast generation
        batch_size, set_size, *data_shape = samples_s.shape
        # TODO: make sure this reshaping is correct.
        samples_s = samples_s.reshape(-1, *data_shape)
        
        # Generate forecast
        forecast = gen.sample(samples_s, enc_s, enc_t)
        
        # Convert to numpy and structure like snapMMD forecast
        forecast_np = forecast.detach().cpu().numpy()
        
        # Create forecast structure that matches snapMMD format
        # snapMMD format expects shape (n_timesteps, n_particles, n_dims)
        # We only have one timestep for the forecast
        forecast_structured = forecast_np[None, :, :]  # Add time dimension
        
        return {
            'forecast': forecast_structured,
            'X_val': training_data['X_val_true']
        }
        
    def load_data_and_forecast(self):
        """Load training data and forecast results."""
        if self.logger:
            self.logger.info(f"Loading training data for {self.dataset_name}...")
        else:
            print(f"Loading training data for {self.dataset_name}...")
        training_data = np.load(self.config['data_path'])
        
        # TODO: make sure data is loaded correctly.
        # Extract training data components
        N_steps = training_data['N_steps']
        Xs_training = [training_data["Xs"][i] for i in range(N_steps-1)]
        X_val_true = training_data["Xs"][-1]
        dts = training_data['dts']
        y0 = training_data['y0']
        time_scale = training_data['time_scale']
        
        # Create training data structure
        training_data_dict = {
            'N_steps': N_steps,
            'Xs': Xs_training,
            'X_val_true': X_val_true,
            'dts': dts,
            'y0': y0,
            'time_scale': time_scale
        }
        
        # Load or generate forecast based on method
        if self.forecast_method == 'snapMMD':
            if self.logger:
                self.logger.info(f"Loading snapMMD forecast results for {self.dataset_name} with seed {self.seed}...")
            else:
                print(f"Loading snapMMD forecast results for {self.dataset_name} with seed {self.seed}...")
            forecast_data = np.load(f"snapMMD_forecasts/{self.dataset_name}_forecast_{self.seed}.npz")
            
            # Extract forecast results - take only the second element (index 1:)
            # snapMMD forecast has shape (2, N, D), we want (1, N, D)
            forecast = forecast_data['forecast'][1:]  # Take second element only
            X_val_forecast = forecast_data['X_val']
            
        elif self.forecast_method == 'CDE':
            if self.logger:
                self.logger.info(f"Generating CDE forecast for {self.dataset_name}...")
            else:
                print(f"Generating CDE forecast for {self.dataset_name}...")
            cde_forecast_data = self.generate_cde_forecast(training_data_dict)
            
            # Extract forecast results - remove first dimension
            # CDE forecast has shape (1, 1, N, D), we want (1, N, D)
            # TODO: make sure this is still correct.
            forecast = cde_forecast_data['forecast'][0]  # Remove first dimension
            X_val_forecast = cde_forecast_data['X_val']
        
        if self.logger:
            self.logger.info(f"Final forecast shape after normalization: {forecast.shape}")
            self.logger.info(f"Forecast method: {self.forecast_method}")
        else:
            print(f"Final forecast shape after normalization: {forecast.shape}")
            print(f"Forecast method: {self.forecast_method}")
        
        results = {
            'training_data': training_data_dict,
            'forecast_data': {
                'forecast': forecast,
                'X_val_forecast': X_val_forecast
            },
            'metadata': {
                'task_name': self.dataset_name,
                'seed': self.seed,
                'config': self.config,
                'forecast_method': self.forecast_method
            }
        }
        
        if self.logger:
            self.logger.info(f"Successfully loaded data:")
            self.logger.info(f"  - Training sequences: {len(Xs_training)}")
            self.logger.info(f"  - Training data shape: {Xs_training[0].shape}")
            self.logger.info(f"  - Forecast shape: {forecast.shape}")
            self.logger.info(f"  - Time scale: {time_scale}")
        else:
            print(f"Successfully loaded data:")
            print(f"  - Training sequences: {len(Xs_training)}")
            print(f"  - Training data shape: {Xs_training[0].shape}")
            print(f"  - Forecast shape: {forecast.shape}")
            print(f"  - Time scale: {time_scale}")
        
        return results
    
    def setup_pca_if_needed(self):
        """Set up PCA for PBMC dataset."""
        if not self.config.get('requires_pca', False):
            return None
            
        if self.logger:
            self.logger.info("Computing PCA components from both PBMC datasets...")
        else:
            print("Computing PCA components from both PBMC datasets...")
        
        # Load both datasets for PCA computation
        data1 = np.load("data/realdata/processed_pbmc_data_sub500_every_2_until20.npz")
        data2 = np.load("data/realdata/processed_pbmc_data_sub500_every_2_until20_interp_val.npz")
        
        Xs1 = data1["Xs"]
        Xs2 = data2["Xs"]
        
        if self.logger:
            self.logger.info(f"Dataset 1 shape: {Xs1.shape}")
            self.logger.info(f"Dataset 2 shape: {Xs2.shape}")
        else:
            print(f"Dataset 1 shape: {Xs1.shape}")
            print(f"Dataset 2 shape: {Xs2.shape}")
        
        # Verify expected shapes and combine
        if Xs1.shape[0] == 21 and Xs2.shape[0] == 20:
            Xs1, Xs2 = Xs2, Xs1
        
        Xs_combined = np.concatenate([Xs1, Xs2], axis=0)
        if self.logger:
            self.logger.info(f"Combined dataset shape: {Xs_combined.shape}")
        else:
            print(f"Combined dataset shape: {Xs_combined.shape}")
        
        # Reshape for PCA
        n_timepoints, n_cells, n_genes = Xs_combined.shape
        X_reshaped = Xs_combined.reshape(n_timepoints * n_cells, n_genes)
        
        # Fit PCA
        pca = PCA(n_components=3)
        pca.fit(X_reshaped)
        
        if self.logger:
            self.logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
            self.logger.info(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")
        else:
            print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
            print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")
        
        self.pca = pca
        return pca
    
    def get_plot_data(self, data, apply_pca=True):
        """Transform data for plotting (apply PCA if needed)."""
        if self.config.get('requires_pca', False) and apply_pca and self.pca is not None:
            return self.pca.transform(data)
        return data
    
    def get_axes_labels(self):
        """Get appropriate axis labels."""
        if self.config.get('requires_pca', False) and self.pca is not None:
            return [f'PC1 ({self.pca.explained_variance_ratio_[0]:.3f})',
                   f'PC2 ({self.pca.explained_variance_ratio_[1]:.3f})',
                   f'PC3 ({self.pca.explained_variance_ratio_[2]:.3f})']
        return self.config['axes_labels']
    
    def plot_main_results(self, results, output_folder="figures"):
        """Plot main results (handles both 2D and 3D)."""
        # Create nested directory structure: figures/<dataset>_<predictor>_<sampling>_<predictor_loss_weight>/
        if self.predictor_type and self.sampling_type and self.predictor_loss_weight is not None:
            folder_name = f"{self.dataset_name}_{self.predictor_type}_{self.sampling_type}_{self.predictor_loss_weight}"
        elif self.predictor_type and self.sampling_type:
            folder_name = f"{self.dataset_name}_{self.predictor_type}_{self.sampling_type}"
        else:
            folder_name = f"{self.dataset_name}_{self.forecast_method}"
        
        nested_output_folder = os.path.join(output_folder, folder_name)
        os.makedirs(nested_output_folder, exist_ok=True)
        training = results['training_data']
        forecast = results['forecast_data']
        config = results['metadata']['config']
        
        # Set up figure based on dimensionality
        is_3d = config.get('plot_dimensionality', config['dimensionality']) == 3
        
        if is_3d:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        if self.seed is not None:
            fig.suptitle(f"{config['title']} Results (Seed {self.seed})")
        elif 'averaged_over_seeds' in results['metadata']:
            n_seeds = results['metadata']['n_seeds']
            fig.suptitle(f"{config['title']} Results (Average over {n_seeds} seeds)")
        else:
            fig.suptitle(f"{config['title']} Results ({self.forecast_method})")
        
        # Plot training sequences with color progression
        n_sequences = len(training['Xs'])
        cmap = plt.colormaps.get_cmap('coolwarm')  # Use coolwarm colormap for consistency
        
        # Collect all training data for scatter plot with colorbar
        all_training_data = []
        timepoint_colors = []
        
        for i, X in enumerate(training['Xs']):
            X_plot = self.get_plot_data(X)
            all_training_data.append(X_plot)
            timepoint_colors.extend([i+1] * len(X_plot))  # timepoint for each particle
        
        # Flatten training data
        all_training_data = np.concatenate(all_training_data, axis=0)
        
        # Create scatter plot with colorbar
        if is_3d:
            scatter = ax.scatter(all_training_data[:, 0], all_training_data[:, 1], all_training_data[:, 2], 
                              alpha=0.7, s=3.0, c=timepoint_colors, cmap=cmap, vmin=1, vmax=n_sequences)
        else:
            scatter = ax.scatter(all_training_data[:, 0], all_training_data[:, 1], 
                              alpha=0.7, s=3.0, c=timepoint_colors, cmap=cmap, vmin=1, vmax=n_sequences)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Time Point', rotation=270, labelpad=15)
        
        # Plot ground truth and forecast
        true_data = training['X_val_true']
        forecast_data = forecast['forecast'][-1]
        
        true_plot = self.get_plot_data(true_data)
        forecast_plot = self.get_plot_data(forecast_data)
        
        if is_3d:
            ax.scatter(true_plot[:, 0], true_plot[:, 1], true_plot[:, 2],
                      alpha=0.9, s=8.0, color='darkgreen', label='Ground Truth', 
                      marker='o', edgecolor='white', linewidth=0.5)
            ax.scatter(forecast_plot[:, 0], forecast_plot[:, 1], forecast_plot[:, 2],
                      alpha=0.9, s=8.0, color='darkorange', label='Forecast',
                      marker='s', edgecolor='white', linewidth=0.5)
            ax.set_zlabel(self.get_axes_labels()[2])
        else:
            ax.scatter(true_plot[:, 0], true_plot[:, 1],
                      alpha=0.9, s=8.0, color='darkgreen', label='Ground Truth',
                      marker='o', edgecolor='white', linewidth=0.5)
            ax.scatter(forecast_plot[:, 0], forecast_plot[:, 1],
                      alpha=0.9, s=8.0, color='darkorange', label='Forecast',
                      marker='s', edgecolor='white', linewidth=0.5)
        
        # Set labels and title
        axes_labels = self.get_axes_labels()
        ax.set_xlabel(axes_labels[0])
        ax.set_ylabel(axes_labels[1])
        ax.set_title('Training Data, Ground Truth & Forecast Phase Portrait')
        ax.legend()
        if not is_3d:
            ax.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        if self.predictor_type and self.sampling_type and self.predictor_loss_weight is not None:
            filename = f"{self.dataset_name}_{self.predictor_type}_{self.sampling_type}_{self.predictor_loss_weight}_results_{self.forecast_method}.png"
        elif self.predictor_type and self.sampling_type:
            filename = f"{self.dataset_name}_{self.predictor_type}_{self.sampling_type}_results_{self.forecast_method}.png"
        elif self.seed is not None:
            filename = f"{self.dataset_name}_results_seed_{self.seed}_{self.forecast_method}.png"
        else:
            filename = f"{self.dataset_name}_results_{self.forecast_method}.png"
        filepath = os.path.join(nested_output_folder, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.logger:
            self.logger.info(f"Main figure saved to: {filepath}")
        else:
            print(f"Main figure saved to: {filepath}")
    

    
    def plot_trajectories(self, results, output_folder="figures"):
        """Plot individual particle/cell trajectories - creates two separate plots."""
        # Create nested directory structure: figures/<dataset>_<predictor>_<sampling>_<predictor_loss_weight>/
        if self.predictor_type and self.sampling_type and self.predictor_loss_weight is not None:
            folder_name = f"{self.dataset_name}_{self.predictor_type}_{self.sampling_type}_{self.predictor_loss_weight}"
        elif self.predictor_type and self.sampling_type:
            folder_name = f"{self.dataset_name}_{self.predictor_type}_{self.sampling_type}"
        else:
            folder_name = f"{self.dataset_name}_{self.forecast_method}"
        
        nested_output_folder = os.path.join(output_folder, folder_name)
        os.makedirs(nested_output_folder, exist_ok=True)
        training = results['training_data']
        forecast = results['forecast_data']
        config = results['metadata']['config']
        
        is_3d = config.get('plot_dimensionality', config['dimensionality']) == 3
        
        # Get data
        Xs = training['Xs']
        true_data = training['X_val_true']
        forecast_data = forecast['forecast'][-1]
        
        true_plot = self.get_plot_data(true_data)
        forecast_plot = self.get_plot_data(forecast_data)
        
        n_particles = Xs[0].shape[0]
        n_timesteps = len(Xs)
        
        # Plot 1: Training data connected to forecast (no ground truth)
        if is_3d:
            fig1 = plt.figure(figsize=(12, 8))
            ax1 = fig1.add_subplot(111, projection='3d')
        else:
            fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
        
        if self.seed is not None:
            fig1.suptitle(f"{config['title']} Trajectories to Forecast (Seed {self.seed})")
        elif 'averaged_over_seeds' in results['metadata']:
            n_seeds = results['metadata']['n_seeds']
            fig1.suptitle(f"{config['title']} Trajectories to Forecast (Average over {n_seeds} seeds)")
        else:
            fig1.suptitle(f"{config['title']} Trajectories to Forecast ({self.forecast_method})")
        
        # Plot trajectories connected to forecast
        for particle_idx in range(n_particles):
            trajectory_coords = [[] for _ in range(3 if is_3d else 2)]
            
            # Collect positions across training time steps
            for t in range(n_timesteps):
                pos_plot = self.get_plot_data(Xs[t][particle_idx:particle_idx+1, :])
                
                # Ensure we have the right dimensionality
                if pos_plot.shape[1] < len(trajectory_coords):
                    if self.logger:
                        self.logger.warning(f"pos_plot has shape {pos_plot.shape}, expected at least {len(trajectory_coords)} dimensions")
                    else:
                        print(f"Warning: pos_plot has shape {pos_plot.shape}, expected at least {len(trajectory_coords)} dimensions")
                    continue
                    
                for coord_idx in range(len(trajectory_coords)):
                    coord_val = pos_plot[0, coord_idx]
                    # Ensure we're appending a scalar value
                    if hasattr(coord_val, 'item'):
                        coord_val = coord_val.item()
                    trajectory_coords[coord_idx].append(coord_val)
            
            # Add forecast endpoint
            if forecast_plot.shape[1] < len(trajectory_coords):
                if self.logger:
                    self.logger.warning(f"forecast_plot has shape {forecast_plot.shape}, expected at least {len(trajectory_coords)} dimensions")
                else:
                    print(f"Warning: forecast_plot has shape {forecast_plot.shape}, expected at least {len(trajectory_coords)} dimensions")
                continue
                
            for coord_idx in range(len(trajectory_coords)):
                coord_val = forecast_plot[particle_idx, coord_idx]
                # Ensure we're appending a scalar value
                if hasattr(coord_val, 'item'):
                    coord_val = coord_val.item()
                trajectory_coords[coord_idx].append(coord_val)
            
            # Skip plotting if we don't have enough data points
            if len(trajectory_coords[0]) < 2:
                continue
                
            # Plot trajectory
            if is_3d:
                ax1.plot(trajectory_coords[0], trajectory_coords[1], trajectory_coords[2],
                        color='lightgrey', alpha=0.7, linewidth=0.5, zorder=1)
            else:
                ax1.plot(trajectory_coords[0], trajectory_coords[1],
                        color='lightgrey', alpha=0.7, linewidth=0.5, zorder=1)
        
        # Plot colored training time points
        n_sequences = len(training['Xs'])
        cmap = plt.colormaps.get_cmap('coolwarm')  # Use coolwarm colormap for consistency
        
        # Collect all training data for scatter plot with colorbar
        all_training_data = []
        timepoint_colors = []
        
        for i, X in enumerate(training['Xs']):
            X_plot = self.get_plot_data(X)
            all_training_data.append(X_plot)
            timepoint_colors.extend([i+1] * len(X_plot))  # timepoint for each particle
        
        # Flatten training data
        all_training_data = np.concatenate(all_training_data, axis=0)
        
        # Create scatter plot with colorbar
        if is_3d:
            scatter1 = ax1.scatter(all_training_data[:, 0], all_training_data[:, 1], all_training_data[:, 2],
                                 alpha=0.8, s=4.0, c=timepoint_colors, cmap=cmap, vmin=1, vmax=n_sequences, zorder=2)
        else:
            scatter1 = ax1.scatter(all_training_data[:, 0], all_training_data[:, 1],
                                 alpha=0.8, s=4.0, c=timepoint_colors, cmap=cmap, vmin=1, vmax=n_sequences, zorder=2)
        
        # Add colorbar
        cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.8, aspect=20)
        cbar1.set_label('Time Point', rotation=270, labelpad=15)
        
        # Plot forecast endpoint only
        if is_3d:
            ax1.scatter(forecast_plot[:, 0], forecast_plot[:, 1], forecast_plot[:, 2],
                       alpha=0.9, s=8.0, color='darkorange', label='Forecast',
                       marker='s', edgecolor='white', linewidth=0.5, zorder=3)
            ax1.set_zlabel(self.get_axes_labels()[2])
        else:
            ax1.scatter(forecast_plot[:, 0], forecast_plot[:, 1],
                       alpha=0.9, s=8.0, color='darkorange', label='Forecast',
                       marker='s', edgecolor='white', linewidth=0.5, zorder=3)
            ax1.grid(True, alpha=0.3)
        
        # Set labels for plot 1
        axes_labels = self.get_axes_labels()
        ax1.set_xlabel(axes_labels[0])
        ax1.set_ylabel(axes_labels[1])
        ax1.set_title('Training Data Connected to Forecast')
        ax1.legend()
        
        plt.tight_layout()
        
        # Save figure 1
        if self.predictor_type and self.sampling_type and self.predictor_loss_weight is not None:
            filename1 = f"{self.dataset_name}_{self.predictor_type}_{self.sampling_type}_{self.predictor_loss_weight}_results_{self.forecast_method}_trajectories_to_forecast.png"
        elif self.predictor_type and self.sampling_type:
            filename1 = f"{self.dataset_name}_{self.predictor_type}_{self.sampling_type}_results_{self.forecast_method}_trajectories_to_forecast.png"
        elif self.seed is not None:
            filename1 = f"{self.dataset_name}_results_seed_{self.seed}_{self.forecast_method}_trajectories_to_forecast.png"
        else:
            filename1 = f"{self.dataset_name}_results_{self.forecast_method}_trajectories_to_forecast.png"
        filepath1 = os.path.join(nested_output_folder, filename1)
        plt.savefig(filepath1, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Training data connected to ground truth (no forecast)
        if is_3d:
            fig2 = plt.figure(figsize=(12, 8))
            ax2 = fig2.add_subplot(111, projection='3d')
        else:
            fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
        
        if self.seed is not None:
            fig2.suptitle(f"{config['title']} Trajectories to Ground Truth (Seed {self.seed})")
        elif 'averaged_over_seeds' in results['metadata']:
            n_seeds = results['metadata']['n_seeds']
            fig2.suptitle(f"{config['title']} Trajectories to Ground Truth (Average over {n_seeds} seeds)")
        else:
            fig2.suptitle(f"{config['title']} Trajectories to Ground Truth ({self.forecast_method})")
        
        # Plot trajectories connected to ground truth
        for particle_idx in range(n_particles):
            trajectory_coords = [[] for _ in range(3 if is_3d else 2)]
            
            # Collect positions across training time steps
            for t in range(n_timesteps):
                pos_plot = self.get_plot_data(Xs[t][particle_idx:particle_idx+1, :])
                
                # Ensure we have the right dimensionality
                if pos_plot.shape[1] < len(trajectory_coords):
                    if self.logger:
                        self.logger.warning(f"pos_plot has shape {pos_plot.shape}, expected at least {len(trajectory_coords)} dimensions")
                    else:
                        print(f"Warning: pos_plot has shape {pos_plot.shape}, expected at least {len(trajectory_coords)} dimensions")
                    continue
                    
                for coord_idx in range(len(trajectory_coords)):
                    coord_val = pos_plot[0, coord_idx]
                    # Ensure we're appending a scalar value
                    if hasattr(coord_val, 'item'):
                        coord_val = coord_val.item()
                    trajectory_coords[coord_idx].append(coord_val)
            
            # Add ground truth endpoint
            if true_plot.shape[1] < len(trajectory_coords):
                if self.logger:
                    self.logger.warning(f"true_plot has shape {true_plot.shape}, expected at least {len(trajectory_coords)} dimensions")
                else:
                    print(f"Warning: true_plot has shape {true_plot.shape}, expected at least {len(trajectory_coords)} dimensions")
                continue
                
            for coord_idx in range(len(trajectory_coords)):
                coord_val = true_plot[particle_idx, coord_idx]
                # Ensure we're appending a scalar value
                if hasattr(coord_val, 'item'):
                    coord_val = coord_val.item()
                trajectory_coords[coord_idx].append(coord_val)
            
            # Skip plotting if we don't have enough data points
            if len(trajectory_coords[0]) < 2:
                continue
                
            # Plot trajectory
            if is_3d:
                ax2.plot(trajectory_coords[0], trajectory_coords[1], trajectory_coords[2],
                        color='lightgrey', alpha=0.7, linewidth=0.5, zorder=1)
            else:
                ax2.plot(trajectory_coords[0], trajectory_coords[1],
                        color='lightgrey', alpha=0.7, linewidth=0.5, zorder=1)
        
        # Plot colored training time points with colorbar
        if is_3d:
            scatter2 = ax2.scatter(all_training_data[:, 0], all_training_data[:, 1], all_training_data[:, 2],
                                 alpha=0.8, s=4.0, c=timepoint_colors, cmap=cmap, vmin=1, vmax=n_sequences, zorder=2)
        else:
            scatter2 = ax2.scatter(all_training_data[:, 0], all_training_data[:, 1],
                                 alpha=0.8, s=4.0, c=timepoint_colors, cmap=cmap, vmin=1, vmax=n_sequences, zorder=2)
        
        # Add colorbar
        cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8, aspect=20)
        cbar2.set_label('Time Point', rotation=270, labelpad=15)
        
        # Plot ground truth endpoint only
        if is_3d:
            ax2.scatter(true_plot[:, 0], true_plot[:, 1], true_plot[:, 2],
                       alpha=0.9, s=8.0, color='darkgreen', label='Ground Truth',
                       marker='o', edgecolor='white', linewidth=0.5, zorder=3)
            ax2.set_zlabel(self.get_axes_labels()[2])
        else:
            ax2.scatter(true_plot[:, 0], true_plot[:, 1],
                       alpha=0.9, s=8.0, color='darkgreen', label='Ground Truth',
                       marker='o', edgecolor='white', linewidth=0.5, zorder=3)
            ax2.grid(True, alpha=0.3)
        
        # Set labels for plot 2
        ax2.set_xlabel(axes_labels[0])
        ax2.set_ylabel(axes_labels[1])
        ax2.set_title('Training Data Connected to Ground Truth')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save figure 2
        if self.predictor_type and self.sampling_type and self.predictor_loss_weight is not None:
            filename2 = f"{self.dataset_name}_{self.predictor_type}_{self.sampling_type}_{self.predictor_loss_weight}_results_{self.forecast_method}_trajectories_to_truth.png"
        elif self.predictor_type and self.sampling_type:
            filename2 = f"{self.dataset_name}_{self.predictor_type}_{self.sampling_type}_results_{self.forecast_method}_trajectories_to_truth.png"
        elif self.seed is not None:
            filename2 = f"{self.dataset_name}_results_seed_{self.seed}_{self.forecast_method}_trajectories_to_truth.png"
        else:
            filename2 = f"{self.dataset_name}_results_{self.forecast_method}_trajectories_to_truth.png"
        filepath2 = os.path.join(nested_output_folder, filename2)
        plt.savefig(filepath2, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.logger:
            self.logger.info(f"Trajectory figures saved:")
            self.logger.info(f"  - To forecast: {filepath1}")
            self.logger.info(f"  - To ground truth: {filepath2}")
        else:
            print(f"Trajectory figures saved:")
            print(f"  - To forecast: {filepath1}")
            print(f"  - To ground truth: {filepath2}")
    
    def plot_interactive_3d(self, results, output_folder="figures"):
        """Create interactive HTML plot for 3D datasets."""
        config = results['metadata']['config']
        if 'interactive_html' not in config['special_plots']:
            return
        
        # Create nested directory structure: figures/<dataset>_<predictor>_<sampling>_<predictor_loss_weight>/
        if self.predictor_type and self.sampling_type and self.predictor_loss_weight is not None:
            folder_name = f"{self.dataset_name}_{self.predictor_type}_{self.sampling_type}_{self.predictor_loss_weight}"
        elif self.predictor_type and self.sampling_type:
            folder_name = f"{self.dataset_name}_{self.predictor_type}_{self.sampling_type}"
        else:
            folder_name = f"{self.dataset_name}_{self.forecast_method}"
        
        nested_output_folder = os.path.join(output_folder, folder_name)
        os.makedirs(nested_output_folder, exist_ok=True)
            
        training = results['training_data']
        forecast = results['forecast_data']
        
        fig = go.Figure()
        
        # Plot training sequences
        n_sequences = len(training['Xs'])
        
        # Combine all training data for continuous colorbar
        all_training_data = []
        timepoint_values = []
        
        for i, X in enumerate(training['Xs']):
            X_plot = self.get_plot_data(X)
            all_training_data.append(X_plot)
            timepoint_values.extend([i+1] * len(X_plot))  # timepoint for each particle
        
        # Flatten training data
        all_training_data = np.concatenate(all_training_data, axis=0)
        
        # Add all training data as single trace with colorbar
        fig.add_trace(go.Scatter3d(
            x=all_training_data[:, 0], 
            y=all_training_data[:, 1], 
            z=all_training_data[:, 2],
            mode='markers',
            marker=dict(
                size=4.0, 
                opacity=0.7, 
                color=timepoint_values,
                colorscale='rdbu',
                colorbar=dict(
                    title="Time Point",
                    thickness=15,
                    len=0.7
                ),
                cmin=1,
                cmax=n_sequences
            ),
            name='Training Data',
            showlegend=False
        ))
        
        # Add ground truth and forecast
        true_data = training['X_val_true']
        forecast_data = forecast['forecast'][-1]
        
        true_plot = self.get_plot_data(true_data)
        forecast_plot = self.get_plot_data(forecast_data)
        
        fig.add_trace(go.Scatter3d(
            x=true_plot[:, 0], y=true_plot[:, 1], z=true_plot[:, 2],
            mode='markers',
            marker=dict(size=10, color='darkgreen', symbol='circle',
                       line=dict(color='white', width=1)),
            name='Ground Truth'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=forecast_plot[:, 0], y=forecast_plot[:, 1], z=forecast_plot[:, 2],
            mode='markers',
            marker=dict(size=10, color='darkorange', symbol='square',
                       line=dict(color='white', width=1)),
            name='Forecast'
        ))
        
        # Set layout
        axes_labels = self.get_axes_labels()
        if self.predictor_type and self.sampling_type and self.predictor_loss_weight is not None:
            title_text = f'{config["title"]} Training Data, Ground Truth & Forecast ({self.predictor_type}, {self.sampling_type}, {self.predictor_loss_weight}) - Interactive'
        elif self.predictor_type and self.sampling_type:
            title_text = f'{config["title"]} Training Data, Ground Truth & Forecast ({self.predictor_type}, {self.sampling_type}) - Interactive'
        elif self.seed is not None:
            title_text = f'{config["title"]} Training Data, Ground Truth & Forecast (Seed {self.seed}) - Interactive'
        else:
            title_text = f'{config["title"]} Training Data, Ground Truth & Forecast ({self.forecast_method}) - Interactive'
        
        fig.update_layout(
            title=title_text,
            scene=dict(
                xaxis_title=axes_labels[0],
                yaxis_title=axes_labels[1],
                zaxis_title=axes_labels[2]
            ),
            width=800,
            height=600
        )
        
        # Save HTML
        if self.predictor_type and self.sampling_type and self.predictor_loss_weight is not None:
            filename = f"{self.dataset_name}_{self.predictor_type}_{self.sampling_type}_{self.predictor_loss_weight}_results_{self.forecast_method}_interactive.html"
        elif self.predictor_type and self.sampling_type:
            filename = f"{self.dataset_name}_{self.predictor_type}_{self.sampling_type}_results_{self.forecast_method}_interactive.html"
        elif self.seed is not None:
            filename = f"{self.dataset_name}_results_seed_{self.seed}_{self.forecast_method}_interactive.html"
        else:
            filename = f"{self.dataset_name}_results_{self.forecast_method}_interactive.html"
        filepath = os.path.join(nested_output_folder, filename)
        fig.write_html(filepath)
        
        if self.logger:
            self.logger.info(f"Interactive figure saved to: {filepath}")
        else:
            print(f"Interactive figure saved to: {filepath}")
    
    def plot_multi_angle_views(self, results, output_folder="figures"):
        """Create front view plot for 3D datasets."""
        config = results['metadata']['config']
        if 'multi_angle' not in config['special_plots']:
            return
        
        # Create nested directory structure: figures/<dataset>_<predictor>_<sampling>_<predictor_loss_weight>/
        if self.predictor_type and self.sampling_type and self.predictor_loss_weight is not None:
            folder_name = f"{self.dataset_name}_{self.predictor_type}_{self.sampling_type}_{self.predictor_loss_weight}"
        elif self.predictor_type and self.sampling_type:
            folder_name = f"{self.dataset_name}_{self.predictor_type}_{self.sampling_type}"
        else:
            folder_name = f"{self.dataset_name}_{self.forecast_method}"
        
        nested_output_folder = os.path.join(output_folder, folder_name)
        os.makedirs(nested_output_folder, exist_ok=True)
            
        training = results['training_data']
        forecast = results['forecast_data']
        
        # Only generate front view
        elev, azim, view_name = 20, 45, "front"
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot training sequences
        n_sequences = len(training['Xs'])
        cmap = plt.colormaps.get_cmap('coolwarm')  # Use coolwarm colormap for consistency
        
        # Collect all training data for scatter plot with colorbar
        all_training_data = []
        timepoint_colors = []
        
        for i, X in enumerate(training['Xs']):
            X_plot = self.get_plot_data(X)
            all_training_data.append(X_plot)
            timepoint_colors.extend([i+1] * len(X_plot))  # timepoint for each particle
        
        # Flatten training data
        all_training_data = np.concatenate(all_training_data, axis=0)
        
        # Create scatter plot with colorbar
        scatter = ax.scatter(all_training_data[:, 0], all_training_data[:, 1], all_training_data[:, 2],
                           alpha=0.7, s=3.0, c=timepoint_colors, cmap=cmap, vmin=1, vmax=n_sequences)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Time Point', rotation=270, labelpad=15)
        
        # Plot endpoints
        true_data = training['X_val_true']
        forecast_data = forecast['forecast'][-1]
        
        true_plot = self.get_plot_data(true_data)
        forecast_plot = self.get_plot_data(forecast_data)
        
        ax.scatter(true_plot[:, 0], true_plot[:, 1], true_plot[:, 2],
                  alpha=0.9, s=8.0, color='darkgreen', label='Ground Truth',
                  marker='o', edgecolor='white', linewidth=0.5)
        ax.scatter(forecast_plot[:, 0], forecast_plot[:, 1], forecast_plot[:, 2],
                  alpha=0.9, s=8.0, color='darkorange', label='Forecast',
                  marker='s', edgecolor='white', linewidth=0.5)
        
        # Set viewing angle and labels
        ax.view_init(elev=elev, azim=azim)
        axes_labels = self.get_axes_labels()
        ax.set_xlabel(axes_labels[0])
        ax.set_ylabel(axes_labels[1])
        ax.set_zlabel(axes_labels[2])
        if self.predictor_type and self.sampling_type and self.predictor_loss_weight is not None:
            ax.set_title(f'{config["title"]} Results ({self.predictor_type}, {self.sampling_type}, {self.predictor_loss_weight}) - {view_name.title()} View')
        elif self.predictor_type and self.sampling_type:
            ax.set_title(f'{config["title"]} Results ({self.predictor_type}, {self.sampling_type}) - {view_name.title()} View')
        elif self.seed is not None:
            ax.set_title(f'{config["title"]} Results (Seed {self.seed}) - {view_name.title()} View')
        else:
            ax.set_title(f'{config["title"]} Results ({self.forecast_method}) - {view_name.title()} View')
        ax.legend()
        
        # Save figure
        if self.predictor_type and self.sampling_type and self.predictor_loss_weight is not None:
            filename = f"{self.dataset_name}_{self.predictor_type}_{self.sampling_type}_{self.predictor_loss_weight}_results_{self.forecast_method}_{view_name}_view.png"
        elif self.predictor_type and self.sampling_type:
            filename = f"{self.dataset_name}_{self.predictor_type}_{self.sampling_type}_results_{self.forecast_method}_{view_name}_view.png"
        elif self.seed is not None:
            filename = f"{self.dataset_name}_results_seed_{self.seed}_{self.forecast_method}_{view_name}_view.png"
        else:
            filename = f"{self.dataset_name}_results_{self.forecast_method}_{view_name}_view.png"
        filepath = os.path.join(nested_output_folder, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.logger:
            self.logger.info(f"Front view saved: {filepath}")
        else:
            print(f"Front view saved: {filepath}")
    
    def plot_individual_final_timepoints(self, results, output_folder="figures"):
        """Plot individual final timepoint plots (PBMC only)."""
        config = results['metadata']['config']
        if 'individual_final' not in config['special_plots']:
            return
        
        # Create nested directory structure: figures/<dataset>_<predictor>_<sampling>_<predictor_loss_weight>/
        if self.predictor_type and self.sampling_type and self.predictor_loss_weight is not None:
            folder_name = f"{self.dataset_name}_{self.predictor_type}_{self.sampling_type}_{self.predictor_loss_weight}"
        elif self.predictor_type and self.sampling_type:
            folder_name = f"{self.dataset_name}_{self.predictor_type}_{self.sampling_type}"
        else:
            folder_name = f"{self.dataset_name}_{self.forecast_method}"
        
        nested_output_folder = os.path.join(output_folder, folder_name)
        os.makedirs(nested_output_folder, exist_ok=True)
            
        training = results['training_data']
        forecast = results['forecast_data']
        
        # Plot 1: Ground Truth only
        fig1 = plt.figure(figsize=(10, 8))
        ax1 = fig1.add_subplot(111, projection='3d')
        
        true_data = training['X_val_true']
        true_plot = self.get_plot_data(true_data)
        
        ax1.scatter(true_plot[:, 0], true_plot[:, 1], true_plot[:, 2],
                   alpha=0.8, s=6.0, color='darkgreen', marker='o',
                   edgecolor='white', linewidth=0.3)
        
        axes_labels = self.get_axes_labels()
        ax1.set_xlabel(axes_labels[0])
        ax1.set_ylabel(axes_labels[1])
        ax1.set_zlabel(axes_labels[2])
        if self.predictor_type and self.sampling_type and self.predictor_loss_weight is not None:
            ax1.set_title(f'{config["title"]} Ground Truth at Final Timepoint ({self.predictor_type}, {self.sampling_type}, {self.predictor_loss_weight})')
        elif self.predictor_type and self.sampling_type:
            ax1.set_title(f'{config["title"]} Ground Truth at Final Timepoint ({self.predictor_type}, {self.sampling_type})')
        elif self.seed is not None:
            ax1.set_title(f'{config["title"]} Ground Truth at Final Timepoint (Seed {self.seed})')
        else:
            ax1.set_title(f'{config["title"]} Ground Truth at Final Timepoint ({self.forecast_method})')
        
        plt.tight_layout()
        
        if self.predictor_type and self.sampling_type and self.predictor_loss_weight is not None:
            filename1 = f"{self.dataset_name}_{self.predictor_type}_{self.sampling_type}_{self.predictor_loss_weight}_results_{self.forecast_method}_ground_truth_only.png"
        elif self.predictor_type and self.sampling_type:
            filename1 = f"{self.dataset_name}_{self.predictor_type}_{self.sampling_type}_results_{self.forecast_method}_ground_truth_only.png"
        elif self.seed is not None:
            filename1 = f"{self.dataset_name}_results_seed_{self.seed}_{self.forecast_method}_ground_truth_only.png"
        else:
            filename1 = f"{self.dataset_name}_results_{self.forecast_method}_ground_truth_only.png"
        filepath1 = os.path.join(nested_output_folder, filename1)
        plt.savefig(filepath1, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Forecast only
        fig2 = plt.figure(figsize=(10, 8))
        ax2 = fig2.add_subplot(111, projection='3d')
        
        forecast_data = forecast['forecast'][-1]
        forecast_plot = self.get_plot_data(forecast_data)
        
        ax2.scatter(forecast_plot[:, 0], forecast_plot[:, 1], forecast_plot[:, 2],
                   alpha=0.8, s=6.0, color='darkorange', marker='s',
                   edgecolor='white', linewidth=0.3)
        
        ax2.set_xlabel(axes_labels[0])
        ax2.set_ylabel(axes_labels[1])
        ax2.set_zlabel(axes_labels[2])
        if self.predictor_type and self.sampling_type and self.predictor_loss_weight is not None:
            ax2.set_title(f'{config["title"]} Forecast at Final Timepoint ({self.predictor_type}, {self.sampling_type}, {self.predictor_loss_weight})')
        elif self.predictor_type and self.sampling_type:
            ax2.set_title(f'{config["title"]} Forecast at Final Timepoint ({self.predictor_type}, {self.sampling_type})')
        elif self.seed is not None:
            ax2.set_title(f'{config["title"]} Forecast at Final Timepoint (Seed {self.seed})')
        else:
            ax2.set_title(f'{config["title"]} Forecast at Final Timepoint ({self.forecast_method})')
        
        plt.tight_layout()
        
        if self.predictor_type and self.sampling_type and self.predictor_loss_weight is not None:
            filename2 = f"{self.dataset_name}_{self.predictor_type}_{self.sampling_type}_{self.predictor_loss_weight}_results_{self.forecast_method}_forecast_only.png"
        elif self.predictor_type and self.sampling_type:
            filename2 = f"{self.dataset_name}_{self.predictor_type}_{self.sampling_type}_results_{self.forecast_method}_forecast_only.png"
        elif self.seed is not None:
            filename2 = f"{self.dataset_name}_results_seed_{self.seed}_{self.forecast_method}_forecast_only.png"
        else:
            filename2 = f"{self.dataset_name}_results_{self.forecast_method}_forecast_only.png"
        filepath2 = os.path.join(nested_output_folder, filename2)
        plt.savefig(filepath2, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.logger:
            self.logger.info(f"Individual final timepoint plots saved:")
            self.logger.info(f"  - Ground truth: {filepath1}")
            self.logger.info(f"  - Forecast: {filepath2}")
        else:
            print(f"Individual final timepoint plots saved:")
            print(f"  - Ground truth: {filepath1}")
            print(f"  - Forecast: {filepath2}")

def calculate_mmd_scores(dataset_name, seeds=[1, 2, 3, 4, 5, 40, 41, 42, 43, 44], forecast_method='snapMMD', cde_forecast_data=None, logger=None):
    """Calculate MMD and MMD^2 scores for multiple seeds (snapMMD) or single run (CDE)."""
    config = DATASET_CONFIGS[dataset_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # TODO: make sure the bandwidth here matches what they use in the paper. It is a bit ambiguous what they meant in the paper by length scale = 1.
    rbf = RBF(bandwidth=2.0).to(device)
    myMMD = MMDLoss(kernel=rbf).to(device)
    
    # Load training data
    if config['experiment_type'] == 'classic':
        training_data = np.load(config['data_path'])
    else:  # realdata
        if dataset_name == 'pbmc':
            training_data = np.load("data/realdata/processed_pbmc_data_sub500_every_2_until20.npz")
        else:
            training_data = np.load(config['data_path'])
    
    N_steps = training_data['N_steps']
    X_val = torch.tensor(training_data["Xs"][-1]).to(device)
    
    mmd_scores = []
    successful_seeds = []
    
    if forecast_method == 'snapMMD':
        if logger:
            logger.info(f"Calculating MMD scores for {dataset_name} across seeds: {seeds}")
        else:
            print(f"Calculating MMD scores for {dataset_name} across seeds: {seeds}")
        
        for seed in seeds:
            try:
                forecast_data = np.load(f"snapMMD_forecasts/{dataset_name}_forecast_{seed}.npz")
                # Take only the second element to match our shape normalization
                forecast = torch.tensor(forecast_data['forecast'][1:]).to(device)
                forecast_final = forecast[-1]
                
                mmd_squared = myMMD(forecast_final, X_val)
                mmd_scores.append(mmd_squared.item())
                successful_seeds.append(seed)
                
                mmd = np.sqrt(mmd_squared.item())
                if logger:
                    logger.info(f"  Seed {seed}: MMD = {mmd:.6f}, MMD^2 = {mmd_squared.item():.6f}")
                else:
                    print(f"  Seed {seed}: MMD = {mmd:.6f}, MMD^2 = {mmd_squared.item():.6f}")
                
            except FileNotFoundError:
                if logger:
                    logger.warning(f"  Seed {seed}: Forecast file not found, skipping...")
                else:
                    print(f"  Seed {seed}: Forecast file not found, skipping...")
                continue
            except Exception as e:
                if logger:
                    logger.error(f"  Seed {seed}: Error - {e}")
                else:
                    print(f"  Seed {seed}: Error - {e}")
                continue
                
    elif forecast_method == 'CDE':
        if logger:
            logger.info(f"Calculating MMD score for {dataset_name} using CDE method")
        else:
            print(f"Calculating MMD score for {dataset_name} using CDE method")
        
        if cde_forecast_data is None:
            if logger:
                logger.error("  CDE forecast data not provided!")
            else:
                print("  Error: CDE forecast data not provided!")
            return None
        
        try:
            forecast = torch.tensor(cde_forecast_data).to(device)
            forecast_final = forecast[-1]
            
            mmd_squared = myMMD(forecast_final, X_val)
            mmd_scores.append(mmd_squared.item())
            successful_seeds.append('CDE')
            
            mmd = np.sqrt(mmd_squared.item())
            if logger:
                logger.info(f"  CDE method: MMD = {mmd:.6f}, MMD^2 = {mmd_squared.item():.6f}")
            else:
                print(f"  CDE method: MMD = {mmd:.6f}, MMD^2 = {mmd_squared.item():.6f}")
            
        except Exception as e:
            if logger:
                logger.error(f"  CDE method: Error - {e}")
            else:
                print(f"  CDE method: Error - {e}")
    
    if not mmd_scores:
        if logger:
            logger.error("No valid MMD^2 scores calculated!")
        else:
            print("No valid MMD^2 scores calculated!")
        return None
    
    mmd_squared_array = np.array(mmd_scores)
    mmd_array = np.sqrt(mmd_squared_array)
    
    if len(mmd_scores) == 1:
        # Single score (CDE case)
        results = {
            'mmd_scores': mmd_array.tolist(),
            'mmd_squared_scores': mmd_scores,
            'seeds': successful_seeds,
            'mean_mmd': mmd_array[0],
            'std_mmd': 0.0,
            'mean_mmd_squared': mmd_scores[0],
            'std_mmd_squared': 0.0,
            'count': 1,
            'task_name': dataset_name,
            'method': forecast_method
        }
    else:
        # Multiple scores (snapMMD case)
        results = {
            'mmd_scores': mmd_array.tolist(),
            'mmd_squared_scores': mmd_scores,
            'seeds': successful_seeds,
            'mean_mmd': np.mean(mmd_array),
            'std_mmd': np.std(mmd_array),
            'mean_mmd_squared': np.mean(mmd_squared_array),
            'std_mmd_squared': np.std(mmd_squared_array),
            'count': len(mmd_scores),
            'task_name': dataset_name,
            'method': forecast_method
        }
    
    return results

def calculate_emd(x, y):
    """Calculate Earth Mover Distance using linear programming."""
    n, m = x.shape[0], y.shape[0]
    
    C = np.linalg.norm(x[:,None] - y[None,:], axis=2).ravel()
    
    A_eq = []
    b_eq = []
    
    for i in range(n):
        row = np.zeros(n*m)
        row[i*m:(i+1)*m] = 1
        A_eq.append(row)
        b_eq.append(1/n)
    
    for j in range(m):
        row = np.zeros(n*m)
        row[j::m] = 1
        A_eq.append(row)
        b_eq.append(1/m)
    
    res = linprog(C, A_eq=np.vstack(A_eq), b_eq=np.array(b_eq), bounds=(0, None), method='highs')
    
    if res.success:
        return res.fun
    else:
        # Note: Cannot access logger here, so still use print for this internal function
        print(f"Warning: EMD optimization failed: {res.message}")
        return np.nan

def calculate_emd_scores(dataset_name, seeds=[1, 2, 3, 4, 5, 40, 41, 42, 43, 44], forecast_method='snapMMD', cde_forecast_data=None, logger=None):
    """Calculate Earth Mover Distance scores for multiple seeds (snapMMD) or single run (CDE)."""
    config = DATASET_CONFIGS[dataset_name]
    
    if not config['calculate_emd']:
        if logger:
            logger.info(f"Skipping EMD calculation for {dataset_name} (disabled in config)")
        else:
            print(f"Skipping EMD calculation for {dataset_name} (disabled in config)")
        return None
    
    # Load training data
    if config['experiment_type'] == 'classic':
        training_data = np.load(config['data_path'])
    else:  # realdata
        if dataset_name == 'pbmc':
            training_data = np.load("data/realdata/processed_pbmc_data_sub500_every_2_until20.npz")
        else:
            training_data = np.load(config['data_path'])
    
    N_steps = training_data['N_steps']
    X_val = training_data["Xs"][-1]
    
    emd_scores = []
    successful_seeds = []
    
    if forecast_method == 'snapMMD':
        if logger:
            logger.info(f"Calculating EMD scores for {dataset_name} across seeds: {seeds}")
        else:
            print(f"Calculating EMD scores for {dataset_name} across seeds: {seeds}")
        
        for seed in seeds:
            try:
                forecast_data = np.load(f"snapMMD_forecasts/{dataset_name}_forecast_{seed}.npz")
                # Take only the second element to match our shape normalization
                forecast = forecast_data['forecast'][1:]
                forecast_final = forecast[-1]
                
                emd = calculate_emd(forecast_final, X_val)
                
                if not np.isnan(emd):
                    emd_scores.append(emd)
                    successful_seeds.append(seed)
                    if logger:
                        logger.info(f"  Seed {seed}: EMD = {emd:.6f}")
                    else:
                        print(f"  Seed {seed}: EMD = {emd:.6f}")
                else:
                    if logger:
                        logger.warning(f"  Seed {seed}: EMD calculation failed")
                    else:
                        print(f"  Seed {seed}: EMD calculation failed")
                    
            except FileNotFoundError:
                if logger:
                    logger.warning(f"  Seed {seed}: Forecast file not found, skipping...")
                else:
                    print(f"  Seed {seed}: Forecast file not found, skipping...")
                continue
            except Exception as e:
                if logger:
                    logger.error(f"  Seed {seed}: Error - {e}")
                else:
                    print(f"  Seed {seed}: Error - {e}")
                continue
                
    elif forecast_method == 'CDE':
        if logger:
            logger.info(f"Calculating EMD score for {dataset_name} using CDE method")
        else:
            print(f"Calculating EMD score for {dataset_name} using CDE method")
        
        if cde_forecast_data is None:
            if logger:
                logger.error("  CDE forecast data not provided!")
            else:
                print("  Error: CDE forecast data not provided!")
            return None
        
        try:
            forecast_final = cde_forecast_data[-1]
            
            emd = calculate_emd(forecast_final, X_val)
            
            if not np.isnan(emd):
                emd_scores.append(emd)
                successful_seeds.append('CDE')
                if logger:
                    logger.info(f"  CDE method: EMD = {emd:.6f}")
                else:
                    print(f"  CDE method: EMD = {emd:.6f}")
            else:
                if logger:
                    logger.warning(f"  CDE method: EMD calculation failed")
                else:
                    print(f"  CDE method: EMD calculation failed")
                
        except Exception as e:
            if logger:
                logger.error(f"  CDE method: Error - {e}")
            else:
                print(f"  CDE method: Error - {e}")
    
    if not emd_scores:
        if logger:
            logger.error("No valid EMD scores calculated!")
        else:
            print("No valid EMD scores calculated!")
        return None
    
    emd_array = np.array(emd_scores)
    
    if len(emd_scores) == 1:
        # Single score (CDE case)
        results = {
            'emd_scores': emd_scores,
            'seeds': successful_seeds,
            'mean_emd': emd_array[0],
            'std_emd': 0.0,
            'count': 1,
            'task_name': dataset_name,
            'method': forecast_method
        }
    else:
        # Multiple scores (snapMMD case)
        results = {
            'emd_scores': emd_scores,
            'seeds': successful_seeds,
            'mean_emd': np.mean(emd_array),
            'std_emd': np.std(emd_array),
            'count': len(emd_scores),
            'task_name': dataset_name,
            'method': forecast_method
        }
    
    return results

def main_hyperparameter_analysis():
    """Main function for processing all hyperparameter combinations."""
    parser = argparse.ArgumentParser(description='Unified results loading and analysis for hyperparameter experiments')
    parser.add_argument('--datasets', nargs='+', choices=list(DATASET_CONFIGS.keys()),
                       default=list(DATASET_CONFIGS.keys()),
                       help='Dataset names to process (default: all datasets)')
    parser.add_argument('--forecast-method', type=str, default='CDE', 
                       choices=['snapMMD', 'CDE'],
                       help='Forecasting method: snapMMD (load pre-computed) or CDE (generate on-the-fly) (default: CDE)')
    parser.add_argument('--output-folder', type=str, default='figures',
                       help='Output folder for figures (default: figures)')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip plotting, only calculate metrics')
    parser.add_argument('--skip-metrics', action='store_true', 
                       help='Skip metrics calculation, only plot')
    parser.add_argument('--experiment-pattern', type=str, default='hyperparam',
                       choices=['hyperparam', 'unified'],
                       help='Experiment directory pattern to search: hyperparam (snapMMD_<dataset>_unified_hyperparam_exp_<hash>) or unified (snapMMD_<dataset>_unified_exp_<hash>) (default: hyperparam)')
    
    args = parser.parse_args()
    

    
    print(f"Processing datasets: {args.datasets}")
    print(f"Using forecasting method: {args.forecast_method}")
    print(f"Using experiment pattern: {args.experiment_pattern}")

    original_cwd = "/orcd/archive/abugoot/001/Projects/paolo/CoupledDistributionEmbeddings/"
    base_outputs_dir = os.path.join(original_cwd, 'outputs')
    
    # Process each dataset
    for dataset_name in args.datasets:
        print(f"\n{'='*80}")
        print(f"PROCESSING DATASET: {dataset_name}")
        print(f"{'='*80}")
        
        # Create a temporary loader to find all experiments for this dataset
        temp_loader = UnifiedResultsLoader(dataset_name, forecast_method=args.forecast_method, experiment_pattern=args.experiment_pattern)
        
        try:
            # Find all experiment directories for this dataset
            experiment_dirs = temp_loader.find_all_experiments_with_hash(dataset_name, base_outputs_dir, args.experiment_pattern)
            print(f"Found {len(experiment_dirs)} experiments for {dataset_name}")
            
            # Group experiments by hyperparameters (excluding seed)
            experiment_groups = temp_loader.group_experiments_by_hyperparameters(experiment_dirs)
            print(f"Grouped into {len(experiment_groups)} unique hyperparameter combinations")
            
            # Process each group of experiments (same hyperparameters, different seeds)
            for hyperparam_key, experiment_group in experiment_groups.items():
                try:
                    # Extract hyperparameters from the key
                    predictor_type, sampling_type, predictor_loss_weight = hyperparam_key
                    seeds = [exp['seed'] for exp in experiment_group]
                    
                    print(f"\n--- Processing {dataset_name}: {predictor_type} + {sampling_type} + {predictor_loss_weight} (seeds: {seeds}) ---")
                    
                    # Create loader with hyperparameters
                    loader = UnifiedResultsLoader(
                        dataset_name,
                        forecast_method=args.forecast_method,
                        predictor_type=predictor_type,
                        sampling_type=sampling_type,
                        predictor_loss_weight=predictor_loss_weight,
                        experiment_pattern=args.experiment_pattern
                    )
                    
                    # Set up logging
                    log_filepath = loader.setup_logging(args.output_folder)
                    
                    loader.logger.info(f"Starting analysis for {dataset_name} with {predictor_type} + {sampling_type} + {predictor_loss_weight}")
                    loader.logger.info(f"Processing {len(experiment_group)} experiments with seeds: {seeds}")
                    loader.logger.info(f"Using {args.forecast_method} forecasting method")
                    
                    # Set up PCA if needed (only once per dataset)
                    loader.setup_pca_if_needed()
                    
                    # Compute average forecast from the experiment group
                    results = loader.compute_average_forecast_from_group(experiment_group, dataset_name)
                    
                    if not args.skip_plots:
                        loader.logger.info(f"Generating plots...")
                        print(f"Generating plots for {predictor_type} + {sampling_type}...")
                        
                        # Main plots
                        loader.plot_main_results(results, args.output_folder)
                        loader.plot_trajectories(results, args.output_folder)
                        
                        # Special plots based on configuration
                        loader.plot_interactive_3d(results, args.output_folder)
                        loader.plot_multi_angle_views(results, args.output_folder)
                        loader.plot_individual_final_timepoints(results, args.output_folder)
                    
                    if not args.skip_metrics:
                        loader.logger.info(f"Logging individual seed results and computing statistics...")
                        print(f"Computing grouped metrics for {predictor_type} + {sampling_type} + {predictor_loss_weight}...")
                        
                        # Extract individual results
                        individual_results = results['individual_results']
                        mmd_scores = individual_results['mmd_scores']
                        emd_scores = individual_results['emd_scores']
                        valid_seeds = individual_results['seeds']
                        
                        # Log individual seed results
                        loader.logger.info(f"Individual MMD^2 scores by seed:")
                        for seed, mmd_score in zip(valid_seeds, mmd_scores):
                            mmd = np.sqrt(mmd_score)
                            loader.logger.info(f"  Seed {seed}: MMD = {mmd:.6f}, MMD^2 = {mmd_score:.6f}")
                            
                        if emd_scores:
                            loader.logger.info(f"Individual EMD scores by seed:")
                            for seed, emd_score in zip(valid_seeds, emd_scores):
                                loader.logger.info(f"  Seed {seed}: EMD = {emd_score:.6f}")
                        
                        # Compute and log statistics
                        if mmd_scores:
                            mmd_squared_array = np.array(mmd_scores)
                            mmd_array = np.sqrt(mmd_squared_array)
                            
                            mean_mmd = np.mean(mmd_array)
                            std_mmd = np.std(mmd_array)
                            mean_mmd_squared = np.mean(mmd_squared_array)
                            std_mmd_squared = np.std(mmd_squared_array)
                            
                            loader.logger.info(f"MMD Statistics (n={len(mmd_scores)}):")
                            loader.logger.info(f"  MMD: {mean_mmd:.6f} ± {std_mmd:.6f}")
                            loader.logger.info(f"  MMD^2: {mean_mmd_squared:.6f} ± {std_mmd_squared:.6f}")
                            print(f"  MMD: {mean_mmd:.6f} ± {std_mmd:.6f}")
                            print(f"  MMD^2: {mean_mmd_squared:.6f} ± {std_mmd_squared:.6f}")
                        
                        if emd_scores:
                            emd_array = np.array(emd_scores)
                            mean_emd = np.mean(emd_array)
                            std_emd = np.std(emd_array)
                            
                            loader.logger.info(f"EMD Statistics (n={len(emd_scores)}):")
                            loader.logger.info(f"  EMD: {mean_emd:.6f} ± {std_emd:.6f}")
                            print(f"  EMD: {mean_emd:.6f} ± {std_emd:.6f}")
                    
                    # Get output folder for this experiment
                    folder_name = f"{dataset_name}_{predictor_type}_{sampling_type}_{predictor_loss_weight}"
                    figures_path = os.path.join(args.output_folder, folder_name)
                    
                    completion_message = f"Analysis complete for {dataset_name} ({predictor_type} + {sampling_type} + {predictor_loss_weight}, {len(valid_seeds)} seeds)"
                    loader.logger.info(completion_message)
                    loader.logger.info(f"Figures saved to: {figures_path}/")
                    loader.logger.info(f"Analysis log saved to: {log_filepath}")
                    
                    print(f"  ✓ {completion_message}")
                    print(f"    Figures: {figures_path}/")
                    print(f"    Log: {log_filepath}")
                    
                except Exception as e:
                    print(f"  ✗ Error processing {dataset_name} ({predictor_type if 'predictor_type' in locals() else 'unknown'} + {sampling_type if 'sampling_type' in locals() else 'unknown'}): {e}")
                    continue
            
        except Exception as e:
            print(f"✗ Error processing dataset {dataset_name}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print("HYPERPARAMETER ANALYSIS COMPLETE")
    print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description='Unified results loading and analysis')
    parser.add_argument('dataset', choices=list(DATASET_CONFIGS.keys()),
                       help='Dataset name')
    # TODO: why do we need a seed for plotting?
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for snapMMD forecast (ignored for CDE method) (default: 42)')
    parser.add_argument('--forecast-method', type=str, default='snapMMD', 
                       choices=['snapMMD', 'CDE'],
                       help='Forecasting method: snapMMD (load pre-computed) or CDE (generate on-the-fly) (default: snapMMD)')
    parser.add_argument('--output-folder', type=str, default='figures',
                       help='Output folder for figures (default: figures)')
    parser.add_argument('--experiment-pattern', type=str, default='hyperparam',
                       choices=['hyperparam', 'unified'],
                       help='Experiment directory pattern to search: hyperparam (snapMMD_<dataset>_unified_hyperparam_exp_<hash>) or unified (snapMMD_<dataset>_unified_exp_<hash>) (default: hyperparam)')

    
    args = parser.parse_args()
    
    
    # Initialize loader
    loader = UnifiedResultsLoader(
        args.dataset, 
        args.seed, 
        args.forecast_method,
        experiment_pattern=args.experiment_pattern
    )
    
    # Set up logging
    log_filepath = loader.setup_logging(args.output_folder)
    
    loader.logger.info(f"Starting analysis for {args.dataset} using {args.forecast_method} method")

    loader.logger.info(f"Log file created at: {log_filepath}")
    
    print(f"Using forecasting method: {args.forecast_method}")
    print(f"Using experiment pattern: {args.experiment_pattern}")

    print(f"Logging to: {log_filepath}")
    
    # Set up PCA if needed
    loader.setup_pca_if_needed()
    
    # Load data
    results = loader.load_data_and_forecast()
    
    if not args.skip_plots:
        loader.logger.info(f"Generating plots for {args.dataset}...")
        print(f"Generating plots for {args.dataset}...")
        
        # Main plots
        loader.plot_main_results(results, args.output_folder)
        loader.plot_trajectories(results, args.output_folder)
        
        # Special plots based on configuration
        loader.plot_interactive_3d(results, args.output_folder)
        loader.plot_multi_angle_views(results, args.output_folder)
        loader.plot_individual_final_timepoints(results, args.output_folder)
    
    if not args.skip_metrics:
        loader.logger.info(f"Calculating metrics for {args.dataset}...")
        print(f"Calculating metrics for {args.dataset}...")
        
        # Extract forecast data for CDE method
        cde_forecast = None
        if args.forecast_method == 'CDE':
            cde_forecast = results['forecast_data']['forecast']
        
        # MMD scores
        loader.logger.info("="*60)
        loader.logger.info("MMD CALCULATION")
        loader.logger.info("="*60)
        print("="*60)
        mmd_results = calculate_mmd_scores(args.dataset, forecast_method=args.forecast_method, cde_forecast_data=cde_forecast, logger=loader.logger)
        
        # EMD scores
        loader.logger.info("="*60)
        loader.logger.info("EMD CALCULATION")
        loader.logger.info("="*60)
        print("="*60)
        emd_results = calculate_emd_scores(args.dataset, forecast_method=args.forecast_method, cde_forecast_data=cde_forecast, logger=loader.logger)
        
        # Display results
        if mmd_results is not None:
            summary_lines = [
                "="*60,
                "MMD RESULTS SUMMARY",
                "="*60,
                f"Task: {mmd_results['task_name']}",
                f"Method: {mmd_results['method']}",
                f"Number of runs processed: {mmd_results['count']}"
            ]
            
            if mmd_results['method'] == 'snapMMD':
                summary_lines.extend([
                    f"Seeds: {mmd_results['seeds']}",
                    f"MMD Score: {mmd_results['mean_mmd']:.6f} ± {mmd_results['std_mmd']:.6f}",
                    f"MMD^2 Score: {mmd_results['mean_mmd_squared']:.6f} ± {mmd_results['std_mmd_squared']:.6f}"
                ])
            else:  # CDE
                summary_lines.extend([
                    f"MMD Score: {mmd_results['mean_mmd']:.6f}",
                    f"MMD^2 Score: {mmd_results['mean_mmd_squared']:.6f}"
                ])
            
            summary_lines.append("="*60)
            
            # Log and print results
            for line in summary_lines:
                loader.logger.info(line)
                print(line)
        
        if emd_results is not None:
            emd_summary_lines = [
                "="*60,
                "EMD RESULTS SUMMARY",
                "="*60,
                f"Task: {emd_results['task_name']}",
                f"Method: {emd_results['method']}",
                f"Number of runs processed: {emd_results['count']}"
            ]
            
            if emd_results['method'] == 'snapMMD':
                emd_summary_lines.extend([
                    f"Seeds: {emd_results['seeds']}",
                    f"EMD Score: {emd_results['mean_emd']:.6f} ± {emd_results['std_emd']:.6f}"
                ])
            else:  # CDE
                emd_summary_lines.append(f"EMD Score: {emd_results['mean_emd']:.6f}")
            
            emd_summary_lines.append("="*60)
            
            # Log and print results
            for line in emd_summary_lines:
                loader.logger.info(line)
                print(line)
    
    completion_message = f"Analysis complete for {args.dataset} using {args.forecast_method} method!"
    if loader.predictor_type and loader.sampling_type:
        folder_name = f"{args.dataset}_{loader.predictor_type}_{loader.sampling_type}"
    else:
        folder_name = f"{args.dataset}_{args.forecast_method}"
    figures_path = os.path.join(args.output_folder, folder_name)
    
    loader.logger.info(completion_message)
    loader.logger.info(f"Figures saved to: {figures_path}/")
    loader.logger.info(f"Analysis log saved to: {log_filepath}")
    
    print(f"\n{completion_message}")
    print(f"Figures saved to: {figures_path}/")
    print(f"Log file saved to: {log_filepath}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--hyperparameter-analysis':
        # Remove the --hyperparameter-analysis flag and call the hyperparameter function
        sys.argv.pop(1)
        main_hyperparameter_analysis()
    else:
        main() 