import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from snapMMD.dls import MMDLoss, RBF
from scipy.optimize import linprog
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.cm as cm

def load_pbmc_data_and_forecast(task_name="pbmc", seed=42):
    """
    Load training data and forecast results for the PBMC task.
    
    Args:
        task_name (str): Name of the task (default: "pbmc")
        seed (int): Random seed used for the forecast (default: 42)
    
    Returns:
        dict: Dictionary containing all loaded data
    """
    
    # Load training data (using special PBMC data file)
    print(f"Loading training data for {task_name}...")
    training_data = np.load(f"data/realdata/processed_pbmc_data_sub500_every_2_until20.npz")
    
    # Load forecast results
    print(f"Loading forecast results for {task_name} with seed {seed}...")
    forecast_data = np.load(f"snapMMD_forecasts/{task_name}_forecast_{seed}.npz")
    
    # Extract training data components
    N_steps = training_data['N_steps']
    Xs_training = [training_data["Xs"][i] for i in range(N_steps-1)]  # training data
    X_val_true = training_data["Xs"][-1]  # true forecasting target
    dts = training_data['dts']
    y0 = training_data['y0']
    time_scale = training_data['time_scale']
    
    # Extract forecast results
    forecast = forecast_data['forecast']
    X_val_forecast = forecast_data['X_val']  # this should match X_val_true
    
    # Package everything into a dictionary
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
            'task_name': task_name,
            'seed': seed
        }
    }
    
    print(f"Successfully loaded data:")
    print(f"  - Training sequences: {len(Xs_training)}")
    print(f"  - Training data shape: {Xs_training[0].shape}")
    print(f"  - Forecast shape: {forecast.shape}")
    print(f"  - Time scale: {time_scale}")
    print(f"  - Initial condition shape: {y0.shape}")
    
    return results

def compute_pca_for_pbmc():
    """
    Compute PCA components using both PBMC datasets for robust dimensionality reduction.
    
    Returns:
        sklearn.decomposition.PCA: Fitted PCA object
    """
    print("Computing PCA components from both PBMC datasets...")
    
    # Load both datasets for PCA computation
    data1 = np.load("data/realdata/processed_pbmc_data_sub500_every_2_until20.npz")
    data2 = np.load("data/realdata/processed_pbmc_data_sub500_every_2_until20_interp_val.npz")
    
    # Extract Xs from both datasets
    Xs1 = data1["Xs"]  # Should be shape [20, N, 30]
    Xs2 = data2["Xs"]  # Should be shape [21, N, 30]
    
    print(f"Dataset 1 shape: {Xs1.shape}")
    print(f"Dataset 2 shape: {Xs2.shape}")
    
    # Verify the expected shapes
    if Xs1.shape[0] == 20 and Xs2.shape[0] == 21:
        print("✓ Dataset shapes match expected: [20, N, 30] and [21, N, 30]")
    elif Xs1.shape[0] == 21 and Xs2.shape[0] == 20:
        print("✓ Dataset shapes match expected: [21, N, 30] and [20, N, 30]")
        # Swap if needed
        Xs1, Xs2 = Xs2, Xs1
    else:
        print(f"⚠️  Unexpected dataset shapes: {Xs1.shape} and {Xs2.shape}")
    
    # Stack the datasets: shape will be [41, N, 30]
    Xs_combined = np.concatenate([Xs1, Xs2], axis=0)
    print(f"Combined dataset shape: {Xs_combined.shape}")
    
    # Reshape for PCA: [41*N, 30] - each row is a 30D point
    n_timepoints, n_cells, n_genes = Xs_combined.shape
    X_reshaped = Xs_combined.reshape(n_timepoints * n_cells, n_genes)
    print(f"Reshaped for PCA: {X_reshaped.shape}")
    
    # Fit PCA
    pca = PCA(n_components=3)
    pca.fit(X_reshaped)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    return pca

def plot_results(results, output_folder="figures"):
    """
    Plot the training data and forecast results in 3D (projected via PCA) and save to folder.
    
    Args:
        results (dict): Results dictionary from load_pbmc_data_and_forecast
        output_folder (str): Folder name to save figures (default: "figures")
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    training = results['training_data']
    forecast = results['forecast_data']
    
    # Compute PCA using both datasets
    pca = compute_pca_for_pbmc()
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot training sequences with color progression (project to 3D)
    n_sequences = len(training['Xs'])
    cmap = cm.get_cmap('viridis')  # Color progression from purple to yellow
    
    for i, X in enumerate(training['Xs']):
        X_3d = pca.transform(X)  # Project to first 3 PCs
        color = cmap(i / (n_sequences - 1))  # Normalize to [0, 1]
        ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], alpha=0.7, s=0.8, color=color,
                  label=f'Training t={i+1}' if i < 3 else None)  # Only label first few
    
    # Add ground truth and forecast with distinct colors (project to 3D)
    true_data = training['X_val_true']
    forecast_data = forecast['forecast'][-1]  # Final forecast point
    
    true_data_3d = pca.transform(true_data)
    forecast_data_3d = pca.transform(forecast_data)
    
    ax.scatter(true_data_3d[:, 0], true_data_3d[:, 1], true_data_3d[:, 2], 
               alpha=0.9, s=4.0, color='darkblue', label='Ground Truth', marker='o',
               edgecolor='white', linewidth=0.5)
    ax.scatter(forecast_data_3d[:, 0], forecast_data_3d[:, 1], forecast_data_3d[:, 2], 
               alpha=0.9, s=4.0, color='crimson', label='Forecast', marker='s',
               edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.3f})')
    ax.set_title(f'PBMC Training Data, Ground Truth & Forecast (PCA Projection, Seed {results["metadata"]["seed"]})')
    ax.legend()
    
    plt.tight_layout()
    
    # Save the static figure
    task_name = results['metadata']['task_name']
    seed = results['metadata']['seed']
    filename = f"{task_name}_results_seed_{seed}.png"
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    print(f"Static figure saved to: {filepath}")
    
    # Create and save interactive 3D figure
    plot_interactive_3d_pbmc(results, pca, output_folder)
    
    # Create and save multiple angle views
    plot_multiple_angles_pbmc(results, pca, output_folder)

def plot_interactive_3d_pbmc(results, pca, output_folder="figures"):
    """
    Create and save an interactive 3D plot (PCA projected) that can be rotated.
    
    Args:
        results (dict): Results dictionary from load_pbmc_data_and_forecast
        pca (sklearn.decomposition.PCA): Fitted PCA object
        output_folder (str): Folder name to save figures (default: "figures")
    """
    training = results['training_data']
    forecast = results['forecast_data']
    
    fig = go.Figure()
    
    # Plot training sequences with color progression (project to 3D)
    n_sequences = len(training['Xs'])
    cmap = cm.get_cmap('viridis')
    
    for i, X in enumerate(training['Xs']):
        X_3d = pca.transform(X)
        color = cmap(i / (n_sequences - 1))
        rgb_color = f'rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})'
        fig.add_trace(go.Scatter3d(
            x=X_3d[:, 0], y=X_3d[:, 1], z=X_3d[:, 2],
            mode='markers',
            marker=dict(size=2.5, opacity=0.7, color=rgb_color),
            name=f'Training t={i+1}' if i < 3 else f'Training t={i+1}',
            showlegend=(i < 3)  # Only show first few in legend
        ))
    
    # Add ground truth with distinct color (project to 3D)
    true_data = training['X_val_true']
    true_data_3d = pca.transform(true_data)
    fig.add_trace(go.Scatter3d(
        x=true_data_3d[:, 0], y=true_data_3d[:, 1], z=true_data_3d[:, 2],
        mode='markers',
        marker=dict(size=6, color='darkblue', symbol='circle',
                   line=dict(color='white', width=1)),
        name='Ground Truth'
    ))
    
    # Add forecast with distinct color (project to 3D)
    forecast_data = forecast['forecast'][-1]  # Final forecast point
    forecast_data_3d = pca.transform(forecast_data)
    fig.add_trace(go.Scatter3d(
        x=forecast_data_3d[:, 0], y=forecast_data_3d[:, 1], z=forecast_data_3d[:, 2],
        mode='markers',
        marker=dict(size=6, color='crimson', symbol='square',
                   line=dict(color='white', width=1)),
        name='Forecast'
    ))
    
    fig.update_layout(
        title=f'PBMC Training Data, Ground Truth & Forecast (PCA Projection, Seed {results["metadata"]["seed"]}) - Interactive',
        scene=dict(
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.3f})',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.3f})',
            zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.3f})'
        ),
        width=800,
        height=600
    )
    
    # Save the interactive figure
    task_name = results['metadata']['task_name']
    seed = results['metadata']['seed']
    html_filename = f"{task_name}_results_seed_{seed}_interactive.html"
    html_filepath = os.path.join(output_folder, html_filename)
    fig.write_html(html_filepath)
    
    print(f"Interactive figure saved to: {html_filepath}")

def plot_multiple_angles_pbmc(results, pca, output_folder="figures"):
    """
    Create and save static 3D plots (PCA projected) from multiple viewing angles.
    VS Code can display these PNG files natively.
    
    Args:
        results (dict): Results dictionary from load_pbmc_data_and_forecast  
        pca (sklearn.decomposition.PCA): Fitted PCA object
        output_folder (str): Folder name to save figures (default: "figures")
    """
    training = results['training_data']
    forecast = results['forecast_data']
    task_name = results['metadata']['task_name']
    seed = results['metadata']['seed']
    
    # Define viewing angles: (elevation, azimuth, name)
    angles = [
        (20, 45, "front"),
        (20, 135, "left"),
        (20, 225, "back"), 
        (20, 315, "right"),
        (90, 0, "top"),
        (-90, 0, "bottom")
    ]
    
    for elev, azim, view_name in angles:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot training sequences with color progression (project to 3D)
        n_sequences = len(training['Xs'])
        cmap = cm.get_cmap('viridis')
        
        for i, X in enumerate(training['Xs']):
            X_3d = pca.transform(X)
            color = cmap(i / (n_sequences - 1))
            ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], alpha=0.7, s=0.8, color=color)
        
        # Add ground truth and forecast with distinct colors (project to 3D)
        true_data = training['X_val_true']
        forecast_data = forecast['forecast'][-1]
        
        true_data_3d = pca.transform(true_data)
        forecast_data_3d = pca.transform(forecast_data)
        
        ax.scatter(true_data_3d[:, 0], true_data_3d[:, 1], true_data_3d[:, 2],
                   alpha=0.9, s=4.0, color='darkblue', label='Ground Truth', marker='o',
                   edgecolor='white', linewidth=0.5)
        ax.scatter(forecast_data_3d[:, 0], forecast_data_3d[:, 1], forecast_data_3d[:, 2],
                   alpha=0.9, s=4.0, color='crimson', label='Forecast', marker='s',
                   edgecolor='white', linewidth=0.5)
        
        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.3f})')
        ax.set_title(f'PBMC Results (PCA, Seed {seed}) - {view_name.title()} View')
        ax.legend()
        
        # Save the figure
        filename = f"{task_name}_results_seed_{seed}_{view_name}_view.png"
        filepath = os.path.join(output_folder, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Multi-angle views saved: {task_name}_results_seed_{seed}_[front/left/back/right/top/bottom]_view.png")

def plot_trajectories(results, output_folder="figures"):
    """
    Plot individual cell trajectories through time in 3D PCA space with connecting lines.
    
    Args:
        results (dict): Results dictionary from load_pbmc_data_and_forecast
        output_folder (str): Folder name to save figures (default: "figures")
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    training = results['training_data']
    forecast = results['forecast_data']
    
    # Compute PCA using both datasets (same as in plot_results)
    pca = compute_pca_for_pbmc()
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories: connect same cell across time steps
    Xs = training['Xs']
    n_cells = Xs[0].shape[0]
    n_timesteps = len(Xs)
    
    # For each cell, draw its trajectory through time
    for cell_idx in range(n_cells):
        trajectory_x = []
        trajectory_y = []
        trajectory_z = []
        
        # Collect positions of this cell across all time steps (in PCA space)
        for t in range(n_timesteps):
            # Project this cell's 30D position to 3D PCA space
            cell_30d = Xs[t][cell_idx:cell_idx+1, :]  # Shape [1, 30]
            cell_3d = pca.transform(cell_30d)  # Shape [1, 3]
            
            trajectory_x.append(cell_3d[0, 0])
            trajectory_y.append(cell_3d[0, 1])
            trajectory_z.append(cell_3d[0, 2])
        
        # Plot trajectory as connected line
        ax.plot(trajectory_x, trajectory_y, trajectory_z, color='lightgrey', alpha=0.7, linewidth=0.5, zorder=1)
    
    # Plot time-colored points on top of trajectories (in PCA space)
    n_sequences = len(training['Xs'])
    cmap = cm.get_cmap('viridis')
    
    for i, X in enumerate(training['Xs']):
        X_3d = pca.transform(X)  # Project to first 3 PCs
        color = cmap(i / (n_sequences - 1))
        ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], alpha=0.8, s=1.2, color=color, zorder=2)
    
    # Add ground truth and forecast with distinct colors (in PCA space)
    true_data = training['X_val_true']
    forecast_data = forecast['forecast'][-1]
    
    true_data_3d = pca.transform(true_data)
    forecast_data_3d = pca.transform(forecast_data)
    
    ax.scatter(true_data_3d[:, 0], true_data_3d[:, 1], true_data_3d[:, 2], 
               alpha=0.9, s=4.0, color='darkblue', label='Ground Truth', marker='o',
               edgecolor='white', linewidth=0.5, zorder=3)
    ax.scatter(forecast_data_3d[:, 0], forecast_data_3d[:, 1], forecast_data_3d[:, 2], 
               alpha=0.9, s=4.0, color='crimson', label='Forecast', marker='s',
               edgecolor='white', linewidth=0.5, zorder=3)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.3f})')
    ax.set_title(f'PBMC Individual Cell Trajectories (PCA Projection, Seed {results["metadata"]["seed"]})')
    ax.legend()
    
    plt.tight_layout()
    
    # Save the trajectory figure
    task_name = results['metadata']['task_name']
    seed = results['metadata']['seed']
    filename = f"{task_name}_results_seed_{seed}_trajectories.png"
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Trajectory figure saved to: {filepath}")

def plot_final_timepoints_individual(results, output_folder="figures"):
    """
    Plot forecast and ground truth separately at the final timepoint in 3D PCA space.
    
    Args:
        results (dict): Results dictionary from load_pbmc_data_and_forecast
        output_folder (str): Folder name to save figures (default: "figures")
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    training = results['training_data']
    forecast = results['forecast_data']
    
    # Compute PCA using both datasets (same as in other plots)
    pca = compute_pca_for_pbmc()
    
    task_name = results['metadata']['task_name']
    seed = results['metadata']['seed']
    
    # Plot 1: Ground Truth only
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    
    true_data = training['X_val_true']
    true_data_3d = pca.transform(true_data)
    
    ax1.scatter(true_data_3d[:, 0], true_data_3d[:, 1], true_data_3d[:, 2], 
               alpha=0.8, s=3.0, color='darkblue', marker='o',
               edgecolor='white', linewidth=0.3)
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    ax1.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.3f})')
    ax1.set_title(f'PBMC Ground Truth at Final Timepoint (PCA Projection, Seed {seed})')
    
    plt.tight_layout()
    
    # Save ground truth plot
    filename1 = f"{task_name}_results_seed_{seed}_ground_truth_only.png"
    filepath1 = os.path.join(output_folder, filename1)
    plt.savefig(filepath1, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Forecast only
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    
    forecast_data = forecast['forecast'][-1]  # Final forecast point
    forecast_data_3d = pca.transform(forecast_data)
    
    ax2.scatter(forecast_data_3d[:, 0], forecast_data_3d[:, 1], forecast_data_3d[:, 2], 
               alpha=0.8, s=3.0, color='crimson', marker='s',
               edgecolor='white', linewidth=0.3)
    
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    ax2.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.3f})')
    ax2.set_title(f'PBMC Forecast at Final Timepoint (PCA Projection, Seed {seed})')
    
    plt.tight_layout()
    
    # Save forecast plot
    filename2 = f"{task_name}_results_seed_{seed}_forecast_only.png"
    filepath2 = os.path.join(output_folder, filename2)
    plt.savefig(filepath2, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Individual final timepoint plots saved:")
    print(f"  - Ground truth: {filepath1}")
    print(f"  - Forecast: {filepath2}")

def calculate_mmd_scores(task_name="pbmc", seeds=[1, 2, 3, 4, 5, 40, 41, 42, 43, 44]):
    """
    Calculate MMD and MMD^2 scores for multiple seeds and return statistics.
    
    Args:
        task_name (str): Name of the task (default: "pbmc")
        seeds (list): List of seeds to process
    
    Returns:
        dict: Dictionary containing MMD and MMD^2 scores and statistics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up MMD loss function with RBF kernel having length scale = 1
    rbf = RBF(bandwidth=2.0).to(device)  # bandwidth=2.0 gives length scale=1 in conventional RBF
    myMMD = MMDLoss(kernel=rbf).to(device)
    
    # Load training data once (same for all seeds)
    training_data = np.load(f"data/realdata/processed_pbmc_data_sub500_every_2_until20.npz")
    N_steps = training_data['N_steps']
    X_val = torch.tensor(training_data["Xs"][-1]).to(device)  # true validation target
    
    mmd_scores = []
    successful_seeds = []
    
    print(f"Calculating MMD scores for {task_name} across seeds: {seeds}")
    
    for seed in seeds:
        try:
            # Load forecast for this seed
            forecast_data = np.load(f"snapMMD_forecasts/{task_name}_forecast_{seed}.npz")
            forecast = torch.tensor(forecast_data['forecast']).to(device)
            forecast_final = forecast[-1]  # Final time point of forecast
            
            # Calculate MMD^2 (what the function returns)
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
    
    # Calculate statistics
    # mmd_scores contains MMD^2 values (what MMDLoss returns)
    mmd_squared_array = np.array(mmd_scores)  # These are MMD^2 values
    mmd_array = np.sqrt(mmd_squared_array)    # Take square root to get MMD values
    
    # Statistics for MMD
    mean_mmd = np.mean(mmd_array)
    std_mmd = np.std(mmd_array)
    
    # Statistics for MMD^2
    mean_mmd_squared = np.mean(mmd_squared_array)
    std_mmd_squared = np.std(mmd_squared_array)
    
    results = {
        'mmd_scores': mmd_array.tolist(),      # MMD values
        'mmd_squared_scores': mmd_scores,      # MMD^2 values
        'seeds': successful_seeds,
        'mean_mmd': mean_mmd,
        'std_mmd': std_mmd,
        'mean_mmd_squared': mean_mmd_squared,
        'std_mmd_squared': std_mmd_squared,
        'count': len(mmd_scores),
        'task_name': task_name
    }
    
    return results

def calculate_emd(x, y):
    """
    Calculate Earth Mover Distance between two point sets using linear programming.
    
    Args:
        x (np.array): First point set of shape (n, d)
        y (np.array): Second point set of shape (m, d)
    
    Returns:
        float: Earth Mover Distance
    """
    n, m = x.shape[0], y.shape[0]
    
    # Calculate cost matrix (L2 distances)
    C = np.linalg.norm(x[:,None] - y[None,:], axis=2).ravel()
    
    # Build equality constraints so each row/col sums to the uniform weights
    A_eq = []
    b_eq = []
    
    # For each source point i: sum_j π_ij = 1/n
    for i in range(n):
        row = np.zeros(n*m)
        row[i*m:(i+1)*m] = 1
        A_eq.append(row)
        b_eq.append(1/n)
    
    # For each target point j: sum_i π_ij = 1/m
    for j in range(m):
        row = np.zeros(n*m)
        row[j::m] = 1
        A_eq.append(row)
        b_eq.append(1/m)
    
    # Solve linear programming problem
    res = linprog(C, A_eq=np.vstack(A_eq), b_eq=np.array(b_eq), bounds=(0, None), method='highs')
    
    if res.success:
        return res.fun
    else:
        print(f"Warning: EMD optimization failed: {res.message}")
        return np.nan

def calculate_emd_scores(task_name="pbmc", seeds=[1, 2, 3, 4, 5, 40, 41, 42, 43, 44]):
    """
    Calculate Earth Mover Distance scores for multiple seeds and return statistics.
    
    Args:
        task_name (str): Name of the task (default: "pbmc")
        seeds (list): List of seeds to process
    
    Returns:
        dict: Dictionary containing EMD scores and statistics
    """
    # Load training data once (same for all seeds)
    training_data = np.load(f"data/realdata/processed_pbmc_data_sub500_every_2_until20.npz")
    N_steps = training_data['N_steps']
    X_val = training_data["Xs"][-1]  # true validation target (numpy array)
    
    emd_scores = []
    successful_seeds = []
    
    print(f"Calculating EMD scores for {task_name} across seeds: {seeds}")
    
    for seed in seeds:
        try:
            # Load forecast for this seed
            forecast_data = np.load(f"snapMMD_forecasts/{task_name}_forecast_{seed}.npz")
            forecast = forecast_data['forecast']
            forecast_final = forecast[-1]  # Final time point of forecast (numpy array)
            
            # Calculate EMD
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
    
    # Calculate statistics
    emd_array = np.array(emd_scores)
    mean_emd = np.mean(emd_array)
    std_emd = np.std(emd_array)
    
    results = {
        'emd_scores': emd_scores,
        'seeds': successful_seeds,
        'mean_emd': mean_emd,
        'std_emd': std_emd,
        'count': len(emd_scores),
        'task_name': task_name
    }
    
    return results

if __name__ == "__main__":
    # Load the data for plotting (using seed 42 as example)
    results = load_pbmc_data_and_forecast(task_name="pbmc", seed=42)
    
    # Print some basic information
    print("\nBasic information:")
    print(f"Number of training sequences: {len(results['training_data']['Xs'])}")
    print(f"Shape of each training sequence: {results['training_data']['Xs'][0].shape}")
    print(f"Forecast shape: {results['forecast_data']['forecast'].shape}")
    print(f"Initial condition shape: {results['training_data']['y0'].shape}")
    print(f"Time scale: {results['training_data']['time_scale']}")
    
    # Save plots to figures folder
    plot_results(results)
    
    # Save trajectory plots
    plot_trajectories(results)
    
    # Save individual final timepoint plots
    plot_final_timepoints_individual(results)
    
    # Calculate MMD and MMD^2 scores across all seeds
    print("\n" + "="*60)
    mmd_results = calculate_mmd_scores(task_name="pbmc", seeds=[1, 2, 3, 4, 5, 40, 41, 42, 43, 44])
    
    # Access individual components
    training_sequences = results['training_data']['Xs']
    true_target = results['training_data']['X_val_true']
    forecast = results['forecast_data']['forecast']
    
    print(f"\nData ready for analysis!")
    print(f"Figures have been saved to the 'figures' folder.")
    print(f"Access training sequences: training_sequences")
    print(f"Access true target: true_target")
    print(f"Access forecast: forecast")
    
    # Display results prominently at the end
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