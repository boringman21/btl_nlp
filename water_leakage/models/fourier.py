"""
Fourier analysis functionality for the leak detection system.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def fourier_approximation(t, y, num_terms=10):
    """
    Computes Fourier approximation of y(t) using num_terms terms.
    
    Args:
        t (np.array): Time values (normalized to 0-2π)
        y (np.array): Signal values
        num_terms (int): Number of terms in the Fourier series
        
    Returns:
        np.array: Approximated signal values
    """
    n = len(y)
    a0 = np.mean(y)
    approx = np.full_like(t, a0)

    for k in range(1, num_terms + 1):
        ak = np.sum(y * np.cos(k * t)) * 2 / n
        bk = np.sum(y * np.sin(k * t)) * 2 / n
        approx += ak * np.cos(k * t) + bk * np.sin(k * t)

    return approx

def plot_fourier_approximation(timestamps, values, label, sensor_id, save_dir=None, color='blue', num_terms=10):
    """
    Plots and saves original data vs. its Fourier approximation.

    Args:
        timestamps (pd.Series): Timestamps for x-axis
        values (np.array): Original signal values
        label (str): Label for the signal (e.g., 'Flow')
        sensor_id (str): Sensor ID
        save_dir (str, optional): Directory to save the plot
        color (str): Color for the original line
        num_terms (int): Number of terms in Fourier approximation
        
    Returns:
        plt.Figure: The matplotlib figure object
    """
    # Create save directory if specified
    if save_dir is not None:
        folder_path = os.path.join(save_dir, str(sensor_id))
        os.makedirs(folder_path, exist_ok=True)
    else:
        folder_path = None
    
    n = len(values)
    t_numeric = np.linspace(0, 2 * np.pi, n)
    approx = fourier_approximation(t_numeric, values, num_terms=num_terms)

    fig = plt.figure(figsize=(12, 4))
    plt.plot(t_numeric, values, label=f'{label} (original)', color=color, alpha=0.6)
    plt.plot(t_numeric, approx, label=f'{label} (Fourier approx)', color='black', linestyle='--')
    plt.title(f'{label} with Fourier Approximation (Sensor {sensor_id})')
    plt.xlabel('Normalized Time')
    plt.ylabel(label)
    plt.grid(True)
    plt.legend()

    # Save plot if folder_path is specified
    if folder_path:
        save_path = os.path.join(folder_path, f'{sensor_id}_{label}_fourier.png')
        plt.savefig(save_path)
        print(f"Fourier approximation plot saved to {save_path}")

    return fig 