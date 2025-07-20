import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from snapMMD.dls import MMDLoss, RBF
from scipy.optimize import linprog
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.cm as cm

def load_repressilator_data_and_forecast(task_name="Repressilator", seed=42):
    """
    Load training data and forecast results for the Repressilator task.
    
    Args:
        task_name (str): Name of the task (default: "Repressilator")
        seed (int): Random seed used for the forecast (default: 42)
    
    Returns:
        dict: Dictionary containing all loaded data
    """
    
    # Load training data
    print(f"Loading training data for {task_name}...")
    training_data = np.load(f"data/classic/{task_name}_data.npz")
    
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
    print(f"  - Initial condition: {y0}")
    
    return results

def plot_results(results, output_folder="figures"):
    """
    Plot the training data and forecast results in 3D and save to folder.
    
    Args:
        results (dict): Results dictionary from load_repressilator_data_and_forecast
        output_folder (str): Folder name to save figures (default: "figures")
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    training = results['training_data']
    forecast = results['forecast_data']
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot training sequences with color progression
    n_sequences = len(training['Xs'])
    cmap = cm.get_cmap('viridis')  # Color progression from purple to yellow
    
    for i, X in enumerate(training['Xs']):
        color = cmap(i / (n_sequences - 1))  # Normalize to [0, 1]
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.7, s=0.8, color=color,
                  label=f'Training t={i+1}' if i < 3 else None)  # Only label first few
    
    # Add ground truth and forecast with distinct colors
    true_data = training['X_val_true']
    forecast_data = forecast['forecast'][-1]  # Final forecast point
    ax.scatter(true_data[:, 0], true_data[:, 1], true_data[:, 2], alpha=0.9, s=4.0, 
                color='darkblue', label='Ground Truth', marker='o', edgecolor='white', linewidth=0.5)
    ax.scatter(forecast_data[:, 0], forecast_data[:, 1], forecast_data[:, 2], alpha=0.9, s=4.0, 
                color='crimson', label='Forecast', marker='s', edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    ax.set_title(f'Repressilator Training Data, Ground Truth & Forecast Phase Portrait (Seed {results["metadata"]["seed"]})')
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
    plot_interactive_3d(results, output_folder)
    
    # Create and save multiple angle views  
    plot_multiple_angles(results, output_folder)

def plot_interactive_3d(results, output_folder="figures"):
    """
    Create and save an interactive 3D plot that can be rotated.
    
    Args:
        results (dict): Results dictionary from load_repressilator_data_and_forecast
        output_folder (str): Folder name to save figures (default: "figures")
    """
    training = results['training_data']
    forecast = results['forecast_data']
    
    fig = go.Figure()
    
    # Plot training sequences with color progression
    n_sequences = len(training['Xs'])
    cmap = cm.get_cmap('viridis')
    
    for i, X in enumerate(training['Xs']):
        color = cmap(i / (n_sequences - 1))
        rgb_color = f'rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})'
        fig.add_trace(go.Scatter3d(
            x=X[:, 0], y=X[:, 1], z=X[:, 2],
            mode='markers',
            marker=dict(size=2.5, opacity=0.7, color=rgb_color),
            name=f'Training t={i+1}' if i < 3 else f'Training t={i+1}',
            showlegend=(i < 3)  # Only show first few in legend
        ))
    
    # Add ground truth with distinct color
    true_data = training['X_val_true']
    fig.add_trace(go.Scatter3d(
        x=true_data[:, 0], y=true_data[:, 1], z=true_data[:, 2],
        mode='markers',
        marker=dict(size=6, color='darkblue', symbol='circle', 
                   line=dict(color='white', width=1)),
        name='Ground Truth'
    ))
    
    # Add forecast with distinct color
    forecast_data = forecast['forecast'][-1]  # Final forecast point
    fig.add_trace(go.Scatter3d(
        x=forecast_data[:, 0], y=forecast_data[:, 1], z=forecast_data[:, 2],
        mode='markers',
        marker=dict(size=6, color='crimson', symbol='square',
                   line=dict(color='white', width=1)),
        name='Forecast'
    ))
    
    fig.update_layout(
        title=f'Repressilator Training Data, Ground Truth & Forecast Phase Portrait (Seed {results["metadata"]["seed"]}) - Interactive',
        scene=dict(
            xaxis_title='X1',
            yaxis_title='X2',
            zaxis_title='X3'
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

def plot_multiple_angles(results, output_folder="figures"):
    """
    Create and save static 3D plots from multiple viewing angles.
    VS Code can display these PNG files natively.
    
    Args:
        results (dict): Results dictionary from load_repressilator_data_and_forecast
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
        
        # Plot training sequences with color progression
        n_sequences = len(training['Xs'])
        cmap = cm.get_cmap('viridis')
        
        for i, X in enumerate(training['Xs']):
            color = cmap(i / (n_sequences - 1))
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.7, s=0.8, color=color)
        
        # Add ground truth and forecast with distinct colors
        true_data = training['X_val_true']
        forecast_data = forecast['forecast'][-1]
        ax.scatter(true_data[:, 0], true_data[:, 1], true_data[:, 2], 
                   alpha=0.9, s=4.0, color='darkblue', label='Ground Truth', marker='o',
                   edgecolor='white', linewidth=0.5)
        ax.scatter(forecast_data[:, 0], forecast_data[:, 1], forecast_data[:, 2], 
                   alpha=0.9, s=4.0, color='crimson', label='Forecast', marker='s',
                   edgecolor='white', linewidth=0.5)
        
        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)
        
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('X3')
        ax.set_title(f'Repressilator Results (Seed {seed}) - {view_name.title()} View')
        ax.legend()
        
        # Save the figure
        filename = f"{task_name}_results_seed_{seed}_{view_name}_view.png"
        filepath = os.path.join(output_folder, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Multi-angle views saved: {task_name}_results_seed_{seed}_[front/left/back/right/top/bottom]_view.png")

def plot_trajectories(results, output_folder="figures"):
    """
    Plot individual particle trajectories through time in 3D with connecting lines.
    
    Args:
        results (dict): Results dictionary from load_repressilator_data_and_forecast
        output_folder (str): Folder name to save figures (default: "figures")
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    training = results['training_data']
    forecast = results['forecast_data']
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories: connect same particle across time steps
    Xs = training['Xs']
    n_particles = Xs[0].shape[0]
    n_timesteps = len(Xs)
    
    # For each particle, draw its trajectory through time
    for particle_idx in range(n_particles):
        trajectory_x = []
        trajectory_y = []
        trajectory_z = []
        
        # Collect positions of this particle across all time steps
        for t in range(n_timesteps):
            trajectory_x.append(Xs[t][particle_idx, 0])
            trajectory_y.append(Xs[t][particle_idx, 1])
            trajectory_z.append(Xs[t][particle_idx, 2])
        
        # Plot trajectory as connected line
        ax.plot(trajectory_x, trajectory_y, trajectory_z, color='lightgrey', alpha=0.7, linewidth=0.5, zorder=1)
    
    # Plot time-colored points on top of trajectories
    n_sequences = len(training['Xs'])
    cmap = cm.get_cmap('viridis')
    
    for i, X in enumerate(training['Xs']):
        color = cmap(i / (n_sequences - 1))
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.8, s=1.2, color=color, zorder=2)
    
    # Add ground truth and forecast with distinct colors
    true_data = training['X_val_true']
    forecast_data = forecast['forecast'][-1]
    ax.scatter(true_data[:, 0], true_data[:, 1], true_data[:, 2], alpha=0.9, s=4.0, 
                color='darkblue', label='Ground Truth', marker='o', edgecolor='white', linewidth=0.5, zorder=3)
    ax.scatter(forecast_data[:, 0], forecast_data[:, 1], forecast_data[:, 2], alpha=0.9, s=4.0, 
                color='crimson', label='Forecast', marker='s', edgecolor='white', linewidth=0.5, zorder=3)
    
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    ax.set_title(f'Repressilator Individual Particle Trajectories (Seed {results["metadata"]["seed"]})')
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

def calculate_mmd_scores(task_name="Repressilator", seeds=[1, 2, 3, 4, 5, 40, 41, 42, 43, 44]):
    """
    Calculate MMD and MMD^2 scores for multiple seeds and return statistics.
    
    Args:
        task_name (str): Name of the task (default: "Repressilator")
        seeds (list): List of seeds to process
    
    Returns:
        dict: Dictionary containing MMD and MMD^2 scores and statistics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up MMD loss function with RBF kernel having length scale = 1
    rbf = RBF(bandwidth=2.0).to(device)  # bandwidth=2.0 gives length scale=1 in conventional RBF
    myMMD = MMDLoss(kernel=rbf).to(device)
    
    # Load training data once (same for all seeds)
    training_data = np.load(f"data/classic/{task_name}_data.npz")
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

def calculate_emd_scores(task_name="Repressilator", seeds=[1, 2, 3, 4, 5, 40, 41, 42, 43, 44]):
    """
    Calculate Earth Mover Distance scores for multiple seeds and return statistics.
    
    Args:
        task_name (str): Name of the task (default: "Repressilator")
        seeds (list): List of seeds to process
    
    Returns:
        dict: Dictionary containing EMD scores and statistics
    """
    # Load training data once (same for all seeds)
    training_data = np.load(f"data/classic/{task_name}_data.npz")
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
    results = load_repressilator_data_and_forecast(task_name="Repressilator", seed=42)
    
    # Print some basic information
    print("\nBasic information:")
    print(f"Number of training sequences: {len(results['training_data']['Xs'])}")
    print(f"Shape of each training sequence: {results['training_data']['Xs'][0].shape}")
    print(f"Forecast shape: {results['forecast_data']['forecast'].shape}")
    print(f"Initial condition (y0): {results['training_data']['y0']}")
    print(f"Time scale: {results['training_data']['time_scale']}")
    
    # Save plots to figures folder
    plot_results(results)
    
    # Save trajectory plots
    plot_trajectories(results)
    
    # Calculate MMD and MMD^2 scores across all seeds
    print("\n" + "="*60)
    mmd_results = calculate_mmd_scores(task_name="Repressilator", seeds=[1, 2, 3, 4, 5, 40, 41, 42, 43, 44])
    
    # Calculate EMD scores across all seeds
    print("\n" + "="*60)
    emd_results = calculate_emd_scores(task_name="Repressilator", seeds=[1, 2, 3, 4, 5, 40, 41, 42, 43, 44])
    
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
    
    if emd_results is not None:
        print("\n" + "="*60)
        print("EMD RESULTS SUMMARY")
        print("="*60)
        print(f"Task: {emd_results['task_name']}")
        print(f"Number of seeds processed: {emd_results['count']}")
        print(f"Seeds: {emd_results['seeds']}")
        print(f"\nEMD Score: {emd_results['mean_emd']:.6f} ± {emd_results['std_emd']:.6f}")
        print("="*60) 