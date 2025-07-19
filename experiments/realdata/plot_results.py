import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import torch
import torchsde
from glob import glob

def plot_forecast_vs_true(task_name="GoM"):
    """
    Plot the forecast vs true data for a given task.
    
    Args:
        task_name (str): Name of the task (e.g., "GoM")
    """
    
    # Find all forecast files for this task
    forecast_pattern = f"snapMMD_forecasts/{task_name}_forecast_*.npz"
    forecast_files = glob(forecast_pattern)
    
    if not forecast_files:
        print(f"No forecast files found for task '{task_name}' in pattern {forecast_pattern}")
        return
    
    # Create figure with subplots for each seed
    n_files = len(forecast_files)
    if n_files == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        axes = [ax]
    else:
        # Arrange subplots in a grid
        n_cols = min(3, n_files)
        n_rows = (n_files + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
    
    for i, forecast_file in enumerate(sorted(forecast_files)):
        # Extract seed from filename
        seed = forecast_file.split('_')[-1].split('.')[0]
        
        # Load data
        data = np.load(forecast_file)
        forecast = data['forecast']
        X_val = data['X_val']
        
        # Get the final forecast point (assuming forecast has shape [time_steps, n_particles, n_dims])
        if forecast.ndim == 3:
            forecast_final = forecast[-1]  # Final time point
        else:
            forecast_final = forecast
            
        # Plot on the appropriate subplot
        ax = axes[i] if len(axes) > 1 else axes[0]
        
        # Scatter plot with different colors for true vs forecast
        ax.scatter(X_val[:, 0], X_val[:, 1], 
                  alpha=0.6, s=30, c='blue', label='True', marker='o')
        ax.scatter(forecast_final[:, 0], forecast_final[:, 1], 
                  alpha=0.6, s=30, c='red', label='Forecast', marker='x')
        
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(f'{task_name} - Seed {seed}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Make axes equal for better visualization
        ax.set_aspect('equal', adjustable='box')
    
    # Hide unused subplots if any
    if len(axes) > n_files:
        for j in range(n_files, len(axes)):
            axes[j].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{task_name}_forecast_vs_true.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'plots/{task_name}_forecast_vs_true.pdf', bbox_inches='tight')
    
    print(f"Plots saved to plots/{task_name}_forecast_vs_true.png and .pdf")
    plt.show()

def plot_full_trajectories(task_name="GoM"):
    """
    Plot full trajectories for raw data and generated data from trained model.
    Creates 5 figures: raw data, generated data, side-by-side, and connected line plots.
    
    Args:
        task_name (str): Name of the task (e.g., "GoM")
    """
    
    # Load the original data
    if "pbmc" in task_name:
        data = np.load(f"data/realdata/processed_pbmc_data_sub500_every_2_until20.npz")
    else:
        data = np.load(f"data/realdata/{task_name}_data.npz")
    
    N_steps = data['N_steps']
    dts = torch.tensor(data['dts'])
    y0 = torch.tensor(data['y0'])
    time_scale = torch.tensor(data['time_scale'])
    
    # Reconstruct full raw trajectory
    # data["Xs"] has N_steps entries, where the last one is the validation data
    raw_trajectory = []
    for i in range(N_steps):
        raw_trajectory.append(data["Xs"][i])
    raw_trajectory = np.array(raw_trajectory)
    
    # Find model files
    model_pattern = f"snapMMD_models/{task_name}_model_*.pt"
    model_files = glob(model_pattern)
    
    if not model_files:
        print(f"No model files found for task '{task_name}' in pattern {model_pattern}")
        return
    
    # Load model architecture (need to import from the main script)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Import model classes
    try:
        sys.path.append('experiments/realdata')
        from models import lamboseen, helmholtz, lamboseendiv
        from snapMMD.booleansde import nninputfun
        import torch.nn as nn
    except ImportError as e:
        print(f"Error importing model classes: {e}")
        print("Make sure you're running from the project root directory")
        return
    
    # Get model architecture based on task
    if "GoM" in task_name:
        mymodel = lamboseendiv(0., 0., -1.5, -1.5, 0., 0., -1.5, 0., .01).to(device)
    elif "pbmc" in task_name:
        torch.manual_seed(0)
        n_gene = 30
        m_vec_guess = 20*torch.ones(n_gene) * 5.
        l_vec_guess = 20*torch.ones(n_gene)
        sigma_vec_guess = np.sqrt(20)*torch.ones(n_gene) * .01
        mlp = nn.Sequential(
            nn.Linear(n_gene, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, n_gene)
        ).to(torch.float64)
        mymodel = nninputfun(m_vec_guess, l_vec_guess, sigma_vec_guess, net=mlp, zero_init=False).to(device)
    
    # Generate trajectories for each seed
    for model_file in sorted(model_files):
        seed = model_file.split('_')[-1].split('.')[0]
        print(f"Generating full trajectory for seed {seed}")
        
        # Load trained model
        mymodel.load_state_dict(torch.load(model_file, map_location=device))
        mymodel.eval()
        
        # Create time points for full trajectory
        # dts contains the actual time points, so we just need to add time 0 and scale
        time_points = dts / time_scale
        
        # Debug: check if time points are strictly increasing
        print(f"Time points: {time_points}")
        print(f"Time scale: {time_scale}")
        print(f"dts: {dts}")
        
        # Ensure time points are strictly increasing
        if not torch.all(time_points[1:] > time_points[:-1]):
            print("Warning: Time points are not strictly increasing. Fixing...")
            # Add small epsilon to ensure strict increasing
            for i in range(1, len(time_points)):
                if time_points[i] <= time_points[i-1]:
                    time_points[i] = time_points[i-1] + 1e-6
        
        print(f"Final time points: {time_points}")
        
        # Generate full trajectory
        with torch.no_grad():
            X_0 = torch.tensor(data["Xs"][0]).to(device)
            generated_trajectory = torchsde.sdeint(mymodel, X_0, time_points.to(device), method='euler')
            generated_trajectory = generated_trajectory.cpu().numpy()
        
        # Find axis ranges for consistent scaling
        all_data = np.concatenate([raw_trajectory.reshape(-1, raw_trajectory.shape[-1]), 
                                  generated_trajectory.reshape(-1, generated_trajectory.shape[-1])])
        x_min, x_max = all_data[:, 0].min(), all_data[:, 0].max()
        y_min, y_max = all_data[:, 1].min(), all_data[:, 1].max()
        
        # Add some padding
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        xlim = [x_min - x_padding, x_max + x_padding]
        ylim = [y_min - y_padding, y_max + y_padding]
        
        # Create plots directory
        os.makedirs('plots', exist_ok=True)
        
        # Figure 1: Raw data only
        plt.figure(figsize=(8, 6))
        for i in range(raw_trajectory.shape[0]):
            plt.scatter(raw_trajectory[i, :, 0], raw_trajectory[i, :, 1], 
                       alpha=0.6, s=30, label=f'Time {i}')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title(f'{task_name} - Raw Data Trajectory (Seed {seed})')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(f'plots/{task_name}_raw_trajectory_seed{seed}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'plots/{task_name}_raw_trajectory_seed{seed}.pdf', bbox_inches='tight')
        plt.close()
        
        # Figure 2: Generated data only
        plt.figure(figsize=(8, 6))
        for i in range(generated_trajectory.shape[0]):
            plt.scatter(generated_trajectory[i, :, 0], generated_trajectory[i, :, 1], 
                       alpha=0.6, s=30, label=f'Time {i}')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title(f'{task_name} - Generated Data Trajectory (Seed {seed})')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(f'plots/{task_name}_generated_trajectory_seed{seed}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'plots/{task_name}_generated_trajectory_seed{seed}.pdf', bbox_inches='tight')
        plt.close()
        
        # Figure 3: Side by side comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Raw data subplot
        for i in range(raw_trajectory.shape[0]):
            ax1.scatter(raw_trajectory[i, :, 0], raw_trajectory[i, :, 1], 
                       alpha=0.6, s=30, label=f'Time {i}')
        ax1.set_xlabel('Dimension 1')
        ax1.set_ylabel('Dimension 2')
        ax1.set_title(f'{task_name} - Raw Data')
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Generated data subplot
        for i in range(generated_trajectory.shape[0]):
            ax2.scatter(generated_trajectory[i, :, 0], generated_trajectory[i, :, 1], 
                       alpha=0.6, s=30, label=f'Time {i}')
        ax2.set_xlabel('Dimension 1')
        ax2.set_ylabel('Dimension 2')
        ax2.set_title(f'{task_name} - Generated Data')
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'plots/{task_name}_comparison_seed{seed}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'plots/{task_name}_comparison_seed{seed}.pdf', bbox_inches='tight')
        plt.close()
        
        # Figure 4: Connected lines for raw data
        plt.figure(figsize=(8, 6))
        # Plot each particle's trajectory as connected lines
        for particle_idx in range(raw_trajectory.shape[1]):
            trajectory_points = raw_trajectory[:, particle_idx, :]
            plt.plot(trajectory_points[:, 0], trajectory_points[:, 1], 
                    alpha=0.7, linewidth=1)
            # Mark start and end points
            plt.scatter(trajectory_points[0, 0], trajectory_points[0, 1], 
                       c='green', s=50, marker='o', alpha=0.8)
            plt.scatter(trajectory_points[-1, 0], trajectory_points[-1, 1], 
                       c='red', s=50, marker='s', alpha=0.8)
        
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title(f'{task_name} - Raw Data Connected Trajectories (Seed {seed})')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.grid(True, alpha=0.3)
        # Add legend for start/end points
        plt.scatter([], [], c='green', s=50, marker='o', label='Start')
        plt.scatter([], [], c='red', s=50, marker='s', label='End')
        plt.legend()
        plt.savefig(f'plots/{task_name}_raw_connected_seed{seed}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'plots/{task_name}_raw_connected_seed{seed}.pdf', bbox_inches='tight')
        plt.close()
        
        # Figure 5: Connected lines for generated data
        plt.figure(figsize=(8, 6))
        # Plot each particle's trajectory as connected lines
        for particle_idx in range(generated_trajectory.shape[1]):
            trajectory_points = generated_trajectory[:, particle_idx, :]
            plt.plot(trajectory_points[:, 0], trajectory_points[:, 1], 
                    alpha=0.7, linewidth=1)
            # Mark start and end points
            plt.scatter(trajectory_points[0, 0], trajectory_points[0, 1], 
                       c='green', s=50, marker='o', alpha=0.8)
            plt.scatter(trajectory_points[-1, 0], trajectory_points[-1, 1], 
                       c='red', s=50, marker='s', alpha=0.8)
        
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title(f'{task_name} - Generated Data Connected Trajectories (Seed {seed})')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.grid(True, alpha=0.3)
        # Add legend for start/end points
        plt.scatter([], [], c='green', s=50, marker='o', label='Start')
        plt.scatter([], [], c='red', s=50, marker='s', label='End')
        plt.legend()
        plt.savefig(f'plots/{task_name}_generated_connected_seed{seed}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'plots/{task_name}_generated_connected_seed{seed}.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"Trajectory plots saved for seed {seed}")

def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <task_name> [--trajectories]")
        print("  task_name: Name of the task (e.g., 'GoM', 'pbmc')")
        print("  --trajectories: Plot full trajectories instead of just final forecast vs true")
        print("\nExamples:")
        print("  python plot_results.py GoM                    # Plot final forecast vs true")
        print("  python plot_results.py GoM --trajectories     # Plot full trajectories")
        return
    
    task_name = sys.argv[1]
    plot_trajectories = "--trajectories" in sys.argv
    
    if plot_trajectories:
        print(f"Plotting full trajectories for task: {task_name}")
        plot_full_trajectories(task_name)
    else:
        print(f"Plotting forecast vs true for task: {task_name}")
        plot_forecast_vs_true(task_name)

if __name__ == '__main__':
    main() 