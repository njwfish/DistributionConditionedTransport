import torchsde
import torch
import numpy as np
from models import LotkaVolterra, repressilator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def simulate_sde(model, initial_condition, time_points, method='euler', **kwargs):
    """
    Simulate a stochastic differential equation using torchsde.
    
    Parameters:
    -----------
    model : nn.Module
        The SDE model (LotkaVolterra or repressilator) that implements f(t, y) and g(t, y) methods
    initial_condition : torch.Tensor
        Initial state of the system, shape (batch_size, state_dim)
    time_points : torch.Tensor
        Time points at which to evaluate the solution, shape (n_time_points,)
    method : str, default='euler'
        Integration method for torchsde.sdeint (e.g., 'euler', 'milstein', 'srk')
    **kwargs : dict
        Additional keyword arguments for torchsde.sdeint
        
    Returns:
    --------
    torch.Tensor
        Simulated trajectory with shape (n_time_points, batch_size, state_dim)
    """
    # Ensure tensors are on the correct device
    initial_condition = initial_condition.to(device)
    time_points = time_points.to(device)
    model = model.to(device)
    
    # Simulate the SDE
    trajectory = torchsde.sdeint(
        model, 
        initial_condition, 
        time_points, 
        method=method,
        **kwargs
    )
    
    return trajectory


def create_model(model_type, **model_params):
    """
    Create and return a model instance.
    
    Parameters:
    -----------
    model_type : str
        Either 'LotkaVolterra' or 'repressilator'
    **model_params : dict
        Parameters for the model constructor
        
    Returns:
    --------
    nn.Module
        The instantiated model
    """
    if model_type.lower() == 'lotkavolterra':
        # Default parameters if not provided
        alpha = model_params.get('alpha', 0.5 * 9)
        beta = model_params.get('beta', 0.1 * 9)
        gamma = model_params.get('gamma', 0.1 * 9)
        delta = model_params.get('delta', 0.02 * 9)
        sigma = model_params.get('sigma', 0.01 * 3)
        return LotkaVolterra(alpha, beta, gamma, delta, sigma).to(device)
        
    elif model_type.lower() == 'repressilator':
        # Default parameters if not provided
        beta = model_params.get('beta', 10.)
        n = model_params.get('n', 1.)
        k = model_params.get('k', 1.)
        gamma = model_params.get('gamma', 10.)
        sigma = model_params.get('sigma', 0.03)
        return repressilator(beta, n, k, gamma, sigma).to(device)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'LotkaVolterra' or 'repressilator'")


def simulate_system(model_type, initial_condition, time_points, method='euler', model_params=None, **kwargs):
    """
    Convenience function to create a model and simulate the SDE in one step.
    
    Parameters:
    -----------
    model_type : str
        Either 'LotkaVolterra' or 'repressilator'
    initial_condition : torch.Tensor or array-like
        Initial state of the system
    time_points : torch.Tensor or array-like
        Time points at which to evaluate the solution
    method : str, default='euler'
        Integration method for torchsde.sdeint
    model_params : dict, optional
        Parameters for the model constructor
    **kwargs : dict
        Additional keyword arguments for torchsde.sdeint
        
    Returns:
    --------
    torch.Tensor
        Simulated trajectory with shape (n_time_points, batch_size, state_dim)
    """
    if model_params is None:
        model_params = {}
        
    # Convert inputs to tensors if needed
    if not isinstance(initial_condition, torch.Tensor):
        initial_condition = torch.tensor(initial_condition, dtype=torch.float32)
    if not isinstance(time_points, torch.Tensor):
        time_points = torch.tensor(time_points, dtype=torch.float32)
        
    # Create model
    model = create_model(model_type, **model_params)
    
    # Simulate
    return simulate_sde(model, initial_condition, time_points, method=method, **kwargs)


if __name__ == "__main__":
    # Example usage
    print("Testing SDE simulation...")
    
    # Example 1: Lotka-Volterra system
    print("\n1. Lotka-Volterra system:")
    lv_initial = torch.tensor([[1.0, 1.0]])  # Shape: (1, 2) - one sample with 2 state variables
    lv_times = torch.linspace(0, 10, 100)
    lv_trajectory = simulate_system('LotkaVolterra', lv_initial, lv_times)
    print(f"LV trajectory shape: {lv_trajectory.shape}")
    
    # Example 2: Repressilator system
    print("\n2. Repressilator system:")
    rep_initial = torch.tensor([[1.0, 1.0, 1.0]])  # Shape: (1, 3) - one sample with 3 state variables
    rep_times = torch.linspace(0, 20, 200)
    rep_trajectory = simulate_system('repressilator', rep_initial, rep_times)
    print(f"Repressilator trajectory shape: {rep_trajectory.shape}")
    
    print("\nSimulation completed successfully!") 