"""
Fourier analysis functionality for the leak detection system.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional, Tuple, Union, List
import pandas as pd


def fourier_approximation(t: np.ndarray, y: np.ndarray, num_terms: int = 10) -> np.ndarray:
    """
    Computes Fourier approximation of y(t) using num_terms terms.
    
    Args:
        t (np.ndarray): Time values (normalized to 0-2Ï€)
        y (np.ndarray): Signal values
        num_terms (int): Number of terms in the Fourier series
        
    Returns:
        np.ndarray: Approximated signal values
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Validate inputs
    if t is None or y is None:
        raise ValueError("Time and signal arrays cannot be None")
        
    if len(t) != len(y):
        raise ValueError(f"Time and signal arrays must have the same length. Got {len(t)} and {len(y)}")
        
    if len(t) == 0:
        raise ValueError("Input arrays cannot be empty")
        
    if num_terms < 1:
        raise ValueError(f"Number of terms must be at least 1. Got {num_terms}")
    
    n = len(y)
    a0 = np.mean(y)
    approx = np.full_like(t, a0)

    # Vectorized calculation for improved performance
    k_values = np.arange(1, num_terms + 1).reshape(-1, 1)
    k_t = k_values * t
    
    # Compute all cosines and sines at once
    cos_kt = np.cos(k_t)
    sin_kt = np.sin(k_t)
    
    # Compute all coefficients at once
    ak = np.sum(y * cos_kt, axis=1) * 2 / n
    bk = np.sum(y * sin_kt, axis=1) * 2 / n
    
    # Apply each term to the approximation
    for k in range(num_terms):
        approx += ak[k] * cos_kt[k] + bk[k] * sin_kt[k]

    return approx


def normalize_time(timestamps: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """
    Normalize timestamps to range [0, 2Ï€] for Fourier analysis.
    
    Args:
        timestamps (pd.Series or np.ndarray): Timestamps to normalize
        
    Returns:
        np.ndarray: Normalized time values
        
    Raises:
        ValueError: If timestamps array is empty or None
    """
    if timestamps is None or len(timestamps) == 0:
        raise ValueError("Timestamps array cannot be empty or None")
    
    # Convert to numpy array if it's a pandas Series
    if isinstance(timestamps, pd.Series):
        if pd.api.types.is_datetime64_any_dtype(timestamps):
            # Convert datetime to numeric (seconds since epoch)
            t_numeric = (timestamps - timestamps.min()).dt.total_seconds().to_numpy()
        else:
            t_numeric = timestamps.to_numpy()
    else:
        t_numeric = timestamps
    
    # Normalize to [0, 2Ï€]
    if len(t_numeric) > 1:
        t_min, t_max = np.min(t_numeric), np.max(t_numeric)
        if t_max > t_min:  # Avoid division by zero
            t_normalized = 2 * np.pi * (t_numeric - t_min) / (t_max - t_min)
        else:
            t_normalized = np.zeros_like(t_numeric)
    else:
        t_normalized = np.zeros_like(t_numeric)
    
    return t_normalized


def plot_fourier_approximation(
    timestamps: Union[pd.Series, np.ndarray],
    values: np.ndarray,
    label: str,
    sensor_id: str,
    save_dir: Optional[str] = None,
    color: str = 'blue',
    num_terms: int = 10,
    figsize: Tuple[int, int] = (12, 4)
) -> Optional[plt.Figure]:
    """
    Plots and saves original data vs. its Fourier approximation.

    Args:
        timestamps (pd.Series or np.ndarray): Timestamps for x-axis
        values (np.ndarray): Original signal values
        label (str): Label for the signal (e.g., 'Flow')
        sensor_id (str): Sensor ID
        save_dir (str, optional): Directory to save the plot
        color (str): Color for the original line
        num_terms (int): Number of terms in Fourier approximation
        figsize (tuple): Figure size as (width, height)
        
    Returns:
        plt.Figure or None: The matplotlib figure object or None if failed
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Validate inputs
    if timestamps is None or values is None:
        raise ValueError("Timestamps and values cannot be None")
        
    if len(timestamps) != len(values):
        raise ValueError(f"Timestamps and values must have the same length. Got {len(timestamps)} and {len(values)}")
        
    if len(timestamps) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    # Create save directory if specified
    if save_dir is not None:
        folder_path = os.path.join(save_dir, str(sensor_id))
        os.makedirs(folder_path, exist_ok=True)
    else:
        folder_path = None
    
    try:
        # Normalize time to [0, 2Ï€]
        t_normalized = normalize_time(timestamps)
        
        # Clean NaN values
        valid_mask = ~np.isnan(values)
        if not np.any(valid_mask):
            print(f"Warning: All values are NaN for {label}")
            return None
            
        t_clean = t_normalized[valid_mask]
        values_clean = values[valid_mask]
        
        # Compute Fourier approximation
        approx = fourier_approximation(t_clean, values_clean, num_terms=num_terms)

        # Create plot
        fig = plt.figure(figsize=figsize)
        plt.plot(t_clean, values_clean, label=f'{label} (original)', color=color, alpha=0.6)
        plt.plot(t_clean, approx, label=f'{label} (Fourier, {num_terms} terms)', color='black', linestyle='--')
        plt.title(f'{label} with Fourier Approximation (Sensor {sensor_id})')
        plt.xlabel('Normalized Time (0 to 2Ï€)')
        plt.ylabel(label)
        plt.grid(True)
        plt.legend()

        # Save plot if folder_path is specified
        if folder_path:
            save_path = os.path.join(folder_path, f'{sensor_id}_{label}_fourier.png')
            plt.savefig(save_path)
            print(f"Fourier approximation plot saved to {save_path}")

        return fig
        
    except Exception as e:
        print(f"Error in Fourier analysis: {str(e)}")
        return None


def analyze_frequency_components(
    values: np.ndarray,
    sampling_rate: float = 1.0,
    plot: bool = True,
    figsize: Tuple[int, int] = (10, 6)
) -> Tuple[np.ndarray, np.ndarray, Optional[plt.Figure]]:
    """
    Analyze the frequency components of a signal using FFT.
    
    Args:
        values (np.ndarray): Signal values
        sampling_rate (float): Sampling rate of the signal
        plot (bool): Whether to plot the frequency spectrum
        figsize (tuple): Figure size if plotting
        
    Returns:
        Tuple of:
            np.ndarray: Frequencies
            np.ndarray: Magnitude spectrum
            plt.Figure or None: The matplotlib figure if plot=True, otherwise None
    """
    # Validate inputs
    if values is None or len(values) == 0:
        raise ValueError("Values array cannot be empty or None")
        
    # Clean NaN values by replacing with zeros
    values_clean = np.nan_to_num(values, nan=0.0)
    
    # Compute FFT
    n = len(values_clean)
    fft_result = np.fft.fft(values_clean)
    magnitude = np.abs(fft_result) / n  # Normalize by array length
    
    # Compute frequencies
    frequencies = np.fft.fftfreq(n, d=1/sampling_rate)
    
    # Get only the first half (positive frequencies)
    positive_freq_idx = frequencies >= 0
    frequencies = frequencies[positive_freq_idx]
    magnitude = magnitude[positive_freq_idx]
    
    # Plot if requested
    fig = None
    if plot:
        fig = plt.figure(figsize=figsize)
        plt.plot(frequencies, magnitude)
        plt.title('Frequency Spectrum')
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude')
        plt.grid(True)
        
        # Highlight dominant frequencies
        threshold = np.max(magnitude) * 0.1  # 10% of max amplitude
        dominant_idx = magnitude > threshold
        if np.any(dominant_idx):
            plt.plot(frequencies[dominant_idx], magnitude[dominant_idx], 'ro')
            
            # Annotate the dominant frequencies
            for i, (freq, mag) in enumerate(zip(frequencies[dominant_idx], magnitude[dominant_idx])):
                if i < 5:  # Only annotate top 5 to avoid clutter
                    plt.annotate(f'{freq:.2f}', (freq, mag), textcoords="offset points", 
                                 xytext=(0,10), ha='center')
    
    return frequencies, magnitude, fig 