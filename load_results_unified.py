import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import os
import sys
import argparse
from snapMMD.dls import MMDLoss, RBF
from scipy.optimize import linprog
from sklearn.decomposition import PCA
import plotly.graph_objects as go

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
    def __init__(self, dataset_name, seed=42):
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Dataset {dataset_name} not supported. Choose from: {list(DATASET_CONFIGS.keys())}")
        
        self.config = DATASET_CONFIGS[dataset_name]
        self.dataset_name = dataset_name
        self.seed = seed
        self.pca = None
        
    def load_data_and_forecast(self):
        """Load training data and forecast results."""
        print(f"Loading training data for {self.dataset_name}...")
        training_data = np.load(self.config['data_path'])
        
        print(f"Loading forecast results for {self.dataset_name} with seed {self.seed}...")
        forecast_data = np.load(f"snapMMD_forecasts/{self.dataset_name}_forecast_{self.seed}.npz")
        
        # Extract training data components
        N_steps = training_data['N_steps']
        Xs_training = [training_data["Xs"][i] for i in range(N_steps-1)]
        X_val_true = training_data["Xs"][-1]
        dts = training_data['dts']
        y0 = training_data['y0']
        time_scale = training_data['time_scale']
        
        # Extract forecast results
        forecast = forecast_data['forecast']
        X_val_forecast = forecast_data['X_val']
        
        results = {
            'training_data': {
                'N_steps': N_steps,
                'Xs': Xs_training,
                'X_val_true': X_val_true,
                'dts': dts,
                'y0': y0,
                'time_scale': time_scale
            },
            'forecast_data': {
                'forecast': forecast,
                'X_val_forecast': X_val_forecast
            },
            'metadata': {
                'task_name': self.dataset_name,
                'seed': self.seed,
                'config': self.config
            }
        }
        
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
            
        print("Computing PCA components from both PBMC datasets...")
        
        # Load both datasets for PCA computation
        data1 = np.load("data/realdata/processed_pbmc_data_sub500_every_2_until20.npz")
        data2 = np.load("data/realdata/processed_pbmc_data_sub500_every_2_until20_interp_val.npz")
        
        Xs1 = data1["Xs"]
        Xs2 = data2["Xs"]
        
        print(f"Dataset 1 shape: {Xs1.shape}")
        print(f"Dataset 2 shape: {Xs2.shape}")
        
        # Verify expected shapes and combine
        if Xs1.shape[0] == 21 and Xs2.shape[0] == 20:
            Xs1, Xs2 = Xs2, Xs1
        
        Xs_combined = np.concatenate([Xs1, Xs2], axis=0)
        print(f"Combined dataset shape: {Xs_combined.shape}")
        
        # Reshape for PCA
        n_timepoints, n_cells, n_genes = Xs_combined.shape
        X_reshaped = Xs_combined.reshape(n_timepoints * n_cells, n_genes)
        
        # Fit PCA
        pca = PCA(n_components=3)
        pca.fit(X_reshaped)
        
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
        os.makedirs(output_folder, exist_ok=True)
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
        
        fig.suptitle(f"{config['title']} Results (Seed {self.seed})")
        
        # Plot training sequences with color progression
        n_sequences = len(training['Xs'])
        cmap = cm.get_cmap('coolwarm')  # Blue to red colormap
        
        for i, X in enumerate(training['Xs']):
            X_plot = self.get_plot_data(X)
            color = cmap(i / (n_sequences - 1))
            
            if is_3d:
                ax.scatter(X_plot[:, 0], X_plot[:, 1], X_plot[:, 2], 
                          alpha=0.7, s=3.0, color=color)
            else:
                ax.scatter(X_plot[:, 0], X_plot[:, 1], alpha=0.7, s=3.0, color=color)
        
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
        filename = f"{self.dataset_name}_results_seed_{self.seed}.png"
        filepath = os.path.join(output_folder, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Main figure saved to: {filepath}")
    

    
    def plot_trajectories(self, results, output_folder="figures"):
        """Plot individual particle/cell trajectories - creates two separate plots."""
        os.makedirs(output_folder, exist_ok=True)
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
        
        fig1.suptitle(f"{config['title']} Trajectories to Forecast (Seed {self.seed})")
        
        # Plot trajectories connected to forecast
        for particle_idx in range(n_particles):
            trajectory_coords = [[] for _ in range(3 if is_3d else 2)]
            
            # Collect positions across training time steps
            for t in range(n_timesteps):
                pos_plot = self.get_plot_data(Xs[t][particle_idx:particle_idx+1, :])
                for coord_idx in range(len(trajectory_coords)):
                    trajectory_coords[coord_idx].append(pos_plot[0, coord_idx])
            
            # Add forecast endpoint
            for coord_idx in range(len(trajectory_coords)):
                trajectory_coords[coord_idx].append(forecast_plot[particle_idx, coord_idx])
            
            # Plot trajectory
            if is_3d:
                ax1.plot(trajectory_coords[0], trajectory_coords[1], trajectory_coords[2],
                        color='lightgrey', alpha=0.7, linewidth=0.5, zorder=1)
            else:
                ax1.plot(trajectory_coords[0], trajectory_coords[1],
                        color='lightgrey', alpha=0.7, linewidth=0.5, zorder=1)
        
        # Plot colored training time points
        n_sequences = len(training['Xs'])
        cmap = cm.get_cmap('coolwarm')  # Blue to red colormap
        
        for i, X in enumerate(training['Xs']):
            X_plot = self.get_plot_data(X)
            color = cmap(i / (n_sequences - 1))
            
            if is_3d:
                ax1.scatter(X_plot[:, 0], X_plot[:, 1], X_plot[:, 2],
                           alpha=0.8, s=4.0, color=color, zorder=2)
            else:
                ax1.scatter(X_plot[:, 0], X_plot[:, 1],
                           alpha=0.8, s=4.0, color=color, zorder=2)
        
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
        filename1 = f"{self.dataset_name}_results_seed_{self.seed}_trajectories_to_forecast.png"
        filepath1 = os.path.join(output_folder, filename1)
        plt.savefig(filepath1, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Training data connected to ground truth (no forecast)
        if is_3d:
            fig2 = plt.figure(figsize=(12, 8))
            ax2 = fig2.add_subplot(111, projection='3d')
        else:
            fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
        
        fig2.suptitle(f"{config['title']} Trajectories to Ground Truth (Seed {self.seed})")
        
        # Plot trajectories connected to ground truth
        for particle_idx in range(n_particles):
            trajectory_coords = [[] for _ in range(3 if is_3d else 2)]
            
            # Collect positions across training time steps
            for t in range(n_timesteps):
                pos_plot = self.get_plot_data(Xs[t][particle_idx:particle_idx+1, :])
                for coord_idx in range(len(trajectory_coords)):
                    trajectory_coords[coord_idx].append(pos_plot[0, coord_idx])
            
            # Add ground truth endpoint
            for coord_idx in range(len(trajectory_coords)):
                trajectory_coords[coord_idx].append(true_plot[particle_idx, coord_idx])
            
            # Plot trajectory
            if is_3d:
                ax2.plot(trajectory_coords[0], trajectory_coords[1], trajectory_coords[2],
                        color='lightgrey', alpha=0.7, linewidth=0.5, zorder=1)
            else:
                ax2.plot(trajectory_coords[0], trajectory_coords[1],
                        color='lightgrey', alpha=0.7, linewidth=0.5, zorder=1)
        
        # Plot colored training time points
        for i, X in enumerate(training['Xs']):
            X_plot = self.get_plot_data(X)
            color = cmap(i / (n_sequences - 1))
            
            if is_3d:
                ax2.scatter(X_plot[:, 0], X_plot[:, 1], X_plot[:, 2],
                           alpha=0.8, s=4.0, color=color, zorder=2)
            else:
                ax2.scatter(X_plot[:, 0], X_plot[:, 1],
                           alpha=0.8, s=4.0, color=color, zorder=2)
        
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
        filename2 = f"{self.dataset_name}_results_seed_{self.seed}_trajectories_to_truth.png"
        filepath2 = os.path.join(output_folder, filename2)
        plt.savefig(filepath2, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Trajectory figures saved:")
        print(f"  - To forecast: {filepath1}")
        print(f"  - To ground truth: {filepath2}")
    
    def plot_interactive_3d(self, results, output_folder="figures"):
        """Create interactive HTML plot for 3D datasets."""
        config = results['metadata']['config']
        if 'interactive_html' not in config['special_plots']:
            return
            
        training = results['training_data']
        forecast = results['forecast_data']
        
        fig = go.Figure()
        
        # Plot training sequences
        n_sequences = len(training['Xs'])
        cmap = cm.get_cmap('coolwarm')  # Blue to red colormap
        
        for i, X in enumerate(training['Xs']):
            X_plot = self.get_plot_data(X)
            color = cmap(i / (n_sequences - 1))
            rgb_color = f'rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})'
            
            fig.add_trace(go.Scatter3d(
                x=X_plot[:, 0], y=X_plot[:, 1], z=X_plot[:, 2],
                mode='markers',
                marker=dict(size=4.0, opacity=0.7, color=rgb_color),
                name=f'Training t={i+1}',
                showlegend=(i < 3)
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
        fig.update_layout(
            title=f'{config["title"]} Training Data, Ground Truth & Forecast (Seed {self.seed}) - Interactive',
            scene=dict(
                xaxis_title=axes_labels[0],
                yaxis_title=axes_labels[1],
                zaxis_title=axes_labels[2]
            ),
            width=800,
            height=600
        )
        
        # Save HTML
        filename = f"{self.dataset_name}_results_seed_{self.seed}_interactive.html"
        filepath = os.path.join(output_folder, filename)
        fig.write_html(filepath)
        
        print(f"Interactive figure saved to: {filepath}")
    
    def plot_multi_angle_views(self, results, output_folder="figures"):
        """Create front view plot for 3D datasets."""
        config = results['metadata']['config']
        if 'multi_angle' not in config['special_plots']:
            return
            
        training = results['training_data']
        forecast = results['forecast_data']
        
        # Only generate front view
        elev, azim, view_name = 20, 45, "front"
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot training sequences
        n_sequences = len(training['Xs'])
        cmap = cm.get_cmap('coolwarm')  # Blue to red colormap
        
        for i, X in enumerate(training['Xs']):
            X_plot = self.get_plot_data(X)
            color = cmap(i / (n_sequences - 1))
            ax.scatter(X_plot[:, 0], X_plot[:, 1], X_plot[:, 2],
                      alpha=0.7, s=3.0, color=color)
        
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
        ax.set_title(f'{config["title"]} Results (Seed {self.seed}) - {view_name.title()} View')
        ax.legend()
        
        # Save figure
        filename = f"{self.dataset_name}_results_seed_{self.seed}_{view_name}_view.png"
        filepath = os.path.join(output_folder, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Front view saved: {filepath}")
    
    def plot_individual_final_timepoints(self, results, output_folder="figures"):
        """Plot individual final timepoint plots (PBMC only)."""
        config = results['metadata']['config']
        if 'individual_final' not in config['special_plots']:
            return
            
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
        ax1.set_title(f'{config["title"]} Ground Truth at Final Timepoint (Seed {self.seed})')
        
        plt.tight_layout()
        
        filename1 = f"{self.dataset_name}_results_seed_{self.seed}_ground_truth_only.png"
        filepath1 = os.path.join(output_folder, filename1)
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
        ax2.set_title(f'{config["title"]} Forecast at Final Timepoint (Seed {self.seed})')
        
        plt.tight_layout()
        
        filename2 = f"{self.dataset_name}_results_seed_{self.seed}_forecast_only.png"
        filepath2 = os.path.join(output_folder, filename2)
        plt.savefig(filepath2, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Individual final timepoint plots saved:")
        print(f"  - Ground truth: {filepath1}")
        print(f"  - Forecast: {filepath2}")

def calculate_mmd_scores(dataset_name, seeds=[1, 2, 3, 4, 5, 40, 41, 42, 43, 44]):
    """Calculate MMD and MMD^2 scores for multiple seeds."""
    config = DATASET_CONFIGS[dataset_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    print(f"Calculating MMD scores for {dataset_name} across seeds: {seeds}")
    
    for seed in seeds:
        try:
            forecast_data = np.load(f"snapMMD_forecasts/{dataset_name}_forecast_{seed}.npz")
            forecast = torch.tensor(forecast_data['forecast']).to(device)
            forecast_final = forecast[-1]
            
            mmd_squared = myMMD(forecast_final, X_val)
            mmd_scores.append(mmd_squared.item())
            successful_seeds.append(seed)
            
            mmd = np.sqrt(mmd_squared.item())
            print(f"  Seed {seed}: MMD = {mmd:.6f}, MMD^2 = {mmd_squared.item():.6f}")
            
        except FileNotFoundError:
            print(f"  Seed {seed}: Forecast file not found, skipping...")
            continue
        except Exception as e:
            print(f"  Seed {seed}: Error - {e}")
            continue
    
    if not mmd_scores:
        print("No valid MMD^2 scores calculated!")
        return None
    
    mmd_squared_array = np.array(mmd_scores)
    mmd_array = np.sqrt(mmd_squared_array)
    
    results = {
        'mmd_scores': mmd_array.tolist(),
        'mmd_squared_scores': mmd_scores,
        'seeds': successful_seeds,
        'mean_mmd': np.mean(mmd_array),
        'std_mmd': np.std(mmd_array),
        'mean_mmd_squared': np.mean(mmd_squared_array),
        'std_mmd_squared': np.std(mmd_squared_array),
        'count': len(mmd_scores),
        'task_name': dataset_name
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
        print(f"Warning: EMD optimization failed: {res.message}")
        return np.nan

def calculate_emd_scores(dataset_name, seeds=[1, 2, 3, 4, 5, 40, 41, 42, 43, 44]):
    """Calculate Earth Mover Distance scores for multiple seeds."""
    config = DATASET_CONFIGS[dataset_name]
    
    if not config['calculate_emd']:
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
    
    print(f"Calculating EMD scores for {dataset_name} across seeds: {seeds}")
    
    for seed in seeds:
        try:
            forecast_data = np.load(f"snapMMD_forecasts/{dataset_name}_forecast_{seed}.npz")
            forecast = forecast_data['forecast']
            forecast_final = forecast[-1]
            
            emd = calculate_emd(forecast_final, X_val)
            
            if not np.isnan(emd):
                emd_scores.append(emd)
                successful_seeds.append(seed)
                print(f"  Seed {seed}: EMD = {emd:.6f}")
            else:
                print(f"  Seed {seed}: EMD calculation failed")
                
        except FileNotFoundError:
            print(f"  Seed {seed}: Forecast file not found, skipping...")
            continue
        except Exception as e:
            print(f"  Seed {seed}: Error - {e}")
            continue
    
    if not emd_scores:
        print("No valid EMD scores calculated!")
        return None
    
    emd_array = np.array(emd_scores)
    
    results = {
        'emd_scores': emd_scores,
        'seeds': successful_seeds,
        'mean_emd': np.mean(emd_array),
        'std_emd': np.std(emd_array),
        'count': len(emd_scores),
        'task_name': dataset_name
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Unified results loading and analysis')
    parser.add_argument('dataset', choices=list(DATASET_CONFIGS.keys()),
                       help='Dataset name')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for forecast (default: 42)')
    parser.add_argument('--output-folder', type=str, default='figures',
                       help='Output folder for figures (default: figures)')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip plotting, only calculate metrics')
    parser.add_argument('--skip-metrics', action='store_true', 
                       help='Skip metrics calculation, only plot')
    
    args = parser.parse_args()
    
    # Initialize loader
    loader = UnifiedResultsLoader(args.dataset, args.seed)
    
    # Set up PCA if needed
    loader.setup_pca_if_needed()
    
    # Load data
    results = loader.load_data_and_forecast()
    
    if not args.skip_plots:
        print(f"\nGenerating plots for {args.dataset}...")
        
        # Main plots
        loader.plot_main_results(results, args.output_folder)
        loader.plot_trajectories(results, args.output_folder)
        
        # Special plots based on configuration
        loader.plot_interactive_3d(results, args.output_folder)
        loader.plot_multi_angle_views(results, args.output_folder)
        loader.plot_individual_final_timepoints(results, args.output_folder)
    
    if not args.skip_metrics:
        print(f"\nCalculating metrics for {args.dataset}...")
        
        # MMD scores
        print("\n" + "="*60)
        mmd_results = calculate_mmd_scores(args.dataset)
        
        # EMD scores
        print("\n" + "="*60)
        emd_results = calculate_emd_scores(args.dataset)
        
        # Display results
        if mmd_results is not None:
            print("\n" + "="*60)
            print("MMD RESULTS SUMMARY")
            print("="*60)
            print(f"Task: {mmd_results['task_name']}")
            print(f"Number of seeds processed: {mmd_results['count']}")
            print(f"Seeds: {mmd_results['seeds']}")
            print(f"\nMMD Score: {mmd_results['mean_mmd']:.6f} ± {mmd_results['std_mmd']:.6f}")
            print(f"MMD^2 Score: {mmd_results['mean_mmd_squared']:.6f} ± {mmd_results['std_mmd_squared']:.6f}")
            print("="*60)
        
        if emd_results is not None:
            print("\n" + "="*60)
            print("EMD RESULTS SUMMARY")
            print("="*60)
            print(f"Task: {emd_results['task_name']}")
            print(f"Number of seeds processed: {emd_results['count']}")
            print(f"Seeds: {emd_results['seeds']}")
            print(f"\nEMD Score: {emd_results['mean_emd']:.6f} ± {emd_results['std_emd']:.6f}")
            print("="*60)
    
    print(f"\nAnalysis complete for {args.dataset}!")
    print(f"Figures saved to: {args.output_folder}/")

if __name__ == "__main__":
    main() 