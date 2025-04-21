"""
Data visualization utilities for the leak detection system.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def _prepare_save_directory(save_dir: Optional[str], sensor_id: str) -> Optional[str]:
    """
    Create and return directory path for saving plots.
    
    Args:
        save_dir (str, optional): Base directory to save plots
        sensor_id (str): ID of the sensor
        
    Returns:
        str or None: Full path to save directory, or None if save_dir is None
    """
    if save_dir is None:
        return None
        
    # Create sensor-specific directory
    folder_path = os.path.join(save_dir, str(sensor_id))
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def _save_plot(fig: plt.Figure, folder_path: Optional[str], filename: str) -> None:
    """
    Save a matplotlib figure to a file if folder_path is provided.
    
    Args:
        fig (plt.Figure): Figure to save
        folder_path (str, optional): Path to save the figure
        filename (str): Filename for the saved figure
    """
    if folder_path:
        save_path = os.path.join(folder_path, filename)
        fig.savefig(save_path)
        print(f"Plot saved to {save_path}")


def _validate_data(df: pd.DataFrame, sensor_id: str, required_cols: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that required columns exist in DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to check
        sensor_id (str): Sensor ID
        required_cols (List[str]): List of required column names
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, missing_columns)
    """
    if df is None or df.empty:
        return False, ["DataFrame is empty"]
        
    # Check for timestamp column
    if 'Timestamp' not in df.columns:
        return False, ["Missing 'Timestamp' column"]
    
    # Check for required sensor columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    return len(missing_cols) == 0, missing_cols


def plot_flow_and_pressure(
    df: pd.DataFrame, 
    sensor_id: str, 
    save_dir: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> Optional[plt.Figure]:
    """
    Plots flow and pressure data from the given DataFrame for the specified sensor ID.
    Optionally saves the plots to the specified directory.
    
    Args:
        df (pd.DataFrame): DataFrame containing sensor data
        sensor_id (str): The sensor ID to plot
        save_dir (str, optional): Directory to save the plots. If None, plots are only displayed.
        figsize (tuple): Figure size as (width, height)
    
    Returns:
        plt.Figure or None: The matplotlib figure object, or None if plotting failed
    """
    # Standardize sensor_id type
    sensor_id = str(sensor_id)
    
    # Define column names
    flow_col = f'{sensor_id}_Flow'
    pressure_1_col = f'{sensor_id}_Pressure_1'
    pressure_2_col = f'{sensor_id}_Pressure_2'
    pressure_diff_col = f'{sensor_id}_Pressure_Diff'
    timestamp_col = 'Timestamp'
    
    # Validate required data
    required_cols = [col for col in [flow_col, pressure_1_col, pressure_2_col] if col in df.columns]
    is_valid, missing_cols = _validate_data(df, sensor_id, [timestamp_col] + required_cols)
    
    if not is_valid:
        print(f"Error: Missing columns in DataFrame: {', '.join(missing_cols)}")
        return None
    
    # Prepare directory for saving
    folder_path = _prepare_save_directory(save_dir, sensor_id)

    # Convert Timestamp to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        try:
            df = df.copy()
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        except Exception as e:
            print(f"Warning: Could not convert timestamps to datetime format: {str(e)}")

    # Initialize the plot
    fig = plt.figure(figsize=figsize)
    
    # Find which metrics are available
    available_metrics = []
    if flow_col in df.columns:
        available_metrics.append(('Flow', flow_col, 'blue'))
    if pressure_1_col in df.columns:
        available_metrics.append(('Pressure_1', pressure_1_col, 'green'))
    if pressure_2_col in df.columns:
        available_metrics.append(('Pressure_2', pressure_2_col, 'red'))
    if pressure_diff_col in df.columns:
        available_metrics.append(('Pressure_Diff', pressure_diff_col, 'purple'))
    elif pressure_1_col in df.columns and pressure_2_col in df.columns:
        # Calculate pressure difference on the fly if not in DataFrame
        available_metrics.append(('Pressure_Diff', None, 'purple'))
    
    # Check if we have any metrics to plot
    if not available_metrics:
        print(f"Error: No metrics available to plot for sensor {sensor_id}")
        plt.close(fig)
        return None
    
    # Plot all available metrics
    for i, (metric_name, col_name, color) in enumerate(available_metrics, 1):
        ax = plt.subplot(len(available_metrics), 1, i)
        
        # Handle pressure difference special case
        if metric_name == 'Pressure_Diff' and col_name is None:
            # Calculate on the fly
            y_values = df[pressure_2_col] - df[pressure_1_col]
            title = f'Pressure Difference (Pressure_2 - Pressure_1) vs Time (Sensor {sensor_id})'
        else:
            y_values = df[col_name]
            title = f'{metric_name} vs Time (Sensor {sensor_id})'
        
        # Plot the data
        ax.plot(df[timestamp_col], y_values, label=metric_name, color=color)
        ax.set_title(title)
        ax.set_xlabel('Timestamp')
        ax.set_ylabel(metric_name)
        ax.grid(True)
        ax.legend()

    # Adjust layout and save if folder_path is specified
    plt.tight_layout()
    if folder_path:
        _save_plot(fig, folder_path, f"sensor_{sensor_id}_plot.png")
    
    return fig


def plot_correlation(
    df: pd.DataFrame, 
    sensor_id: str, 
    save_dir: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> Optional[plt.Figure]:
    """
    Plots the correlation matrix of flow and pressure data for the specified sensor ID.
    
    Args:
        df (pd.DataFrame): DataFrame containing sensor data
        sensor_id (str): The sensor ID to analyze
        save_dir (str, optional): Directory to save the plot. If None, plot is only displayed.
        figsize (tuple): Figure size as (width, height)
    
    Returns:
        plt.Figure or None: The matplotlib figure object, or None if plotting failed
    """
    # Standardize sensor_id type
    sensor_id = str(sensor_id)
    
    # Define column names
    flow_col = f'{sensor_id}_Flow'
    pressure_1_col = f'{sensor_id}_Pressure_1'
    pressure_2_col = f'{sensor_id}_Pressure_2'
    pressure_diff_col = f'{sensor_id}_Pressure_Diff'
    
    # Validate required data
    required_cols = [col for col in [flow_col, pressure_1_col, pressure_2_col] if col in df.columns]
    is_valid, missing_cols = _validate_data(df, sensor_id, required_cols)
    
    if not is_valid or len(required_cols) < 2:
        print(f"Error: Insufficient data for correlation analysis. Need at least 2 metrics.")
        return None
    
    # Prepare directory for saving
    folder_path = _prepare_save_directory(save_dir, sensor_id)

    # Extract relevant columns for correlation
    columns_to_use = [col for col in [flow_col, pressure_1_col, pressure_2_col, pressure_diff_col] 
                     if col in df.columns]
    
    # Select only numeric columns with valid data
    corr_df = df[columns_to_use].select_dtypes(include=['number'])
    
    # Check if we have enough data for correlation
    if corr_df.shape[1] < 2:
        print(f"Error: Need at least 2 numeric columns for correlation matrix.")
        return None
    
    # Calculate correlation matrix
    try:
        corr_matrix = corr_df.corr()
    except Exception as e:
        print(f"Error calculating correlation matrix: {str(e)}")
        return None

    # Plot the correlation matrix as a heatmap
    fig = plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title(f'Correlation Matrix (Sensor {sensor_id})')

    # Save the plot if folder_path is specified
    if folder_path:
        _save_plot(fig, folder_path, f"sensor_{sensor_id}_correlation.png")

    return fig


def plot_all_sensors(
    df: pd.DataFrame, 
    metric: str = 'Flow', 
    save_dir: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
    max_sensors: int = 10
) -> Optional[plt.Figure]:
    """
    Plot the same metric for multiple sensors on a single graph.
    
    Args:
        df (pd.DataFrame): DataFrame with sensor data
        metric (str): Metric to plot ('Flow', 'Pressure_1', 'Pressure_2', or 'Pressure_Diff')
        save_dir (str, optional): Directory to save the plot
        figsize (tuple): Figure size
        max_sensors (int): Maximum number of sensors to include in one plot
        
    Returns:
        plt.Figure or None: The matplotlib figure object, or None if plotting failed
    """
    if df is None or df.empty or 'Timestamp' not in df.columns:
        print("Error: Invalid DataFrame or missing Timestamp column")
        return None
    
    # Find all columns for the specified metric
    cols = [col for col in df.columns if col.endswith(f'_{metric}')]
    
    if not cols:
        print(f"Error: No columns found for metric '{metric}'")
        return None
    
    # Limit the number of sensors to avoid overcrowding
    if len(cols) > max_sensors:
        print(f"Warning: Limiting plot to {max_sensors} sensors (out of {len(cols)} available)")
        cols = cols[:max_sensors]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    for col in cols:
        sensor_id = col.split('_')[0]
        ax.plot(df['Timestamp'], df[col], label=f'Sensor {sensor_id}')
    
    ax.set_title(f'{metric} for Multiple Sensors')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel(metric)
    ax.grid(True)
    ax.legend()
    
    # Save the plot if save_dir is specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        _save_plot(fig, save_dir, f"all_sensors_{metric.lower()}.png")
    
    return fig 


def plot_predictions(
    df: pd.DataFrame, 
    prediction: Dict[str, float], 
    sensor_id: str,
    next_timestamp: pd.Timestamp,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> Optional[plt.Figure]:
    """
    Plot historical data and predicted values for the next 15-minute tick.
    
    Args:
        df (pd.DataFrame): Historical data with 'Timestamp' and sensor columns
        prediction (Dict[str, float]): Dictionary with predictions for each feature
        sensor_id (str): The sensor ID
        next_timestamp (pd.Timestamp): Timestamp for the predicted values
        save_path (str, optional): Path to save the figure
        figsize (tuple): Figure size as (width, height)
    
    Returns:
        plt.Figure or None: The matplotlib figure object, or None if plotting failed
    """
    # Standardize sensor_id
    sensor_id = str(sensor_id)
    
    # Define column names
    flow_col = f'{sensor_id}_Flow'
    pressure_1_col = f'{sensor_id}_Pressure_1'
    pressure_2_col = f'{sensor_id}_Pressure_2'
    timestamp_col = 'Timestamp'
    
    # Validate required data
    required_cols = [timestamp_col]
    metrics_to_plot = []
    
    if flow_col in df.columns and 'Flow' in prediction:
        required_cols.append(flow_col)
        metrics_to_plot.append(('Flow', flow_col, 'blue'))
    
    if pressure_1_col in df.columns and 'Pressure_1' in prediction:
        required_cols.append(pressure_1_col)
        metrics_to_plot.append(('Pressure_1', pressure_1_col, 'green'))
    
    if pressure_2_col in df.columns and 'Pressure_2' in prediction:
        required_cols.append(pressure_2_col)
        metrics_to_plot.append(('Pressure_2', pressure_2_col, 'red'))
    
    # Check if we have data to plot
    if not metrics_to_plot:
        print(f"Error: No metrics available to plot for sensor {sensor_id}")
        return None
    
    # Convert timestamp to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Plot each metric
    for i, (metric_name, col_name, color) in enumerate(metrics_to_plot, 1):
        ax = plt.subplot(len(metrics_to_plot), 1, i)
        
        # Plot historical data
        ax.plot(df[timestamp_col], df[col_name], label=f'Historical {metric_name}', color=color, alpha=0.7)
        
        # Create predicted point - use only last 24 hours of data for better visualization
        last_24h = df[df[timestamp_col] >= df[timestamp_col].max() - pd.Timedelta(hours=24)]
        
        if not last_24h.empty:
            # Plot predicted value
            pred_value = prediction[metric_name]
            ax.scatter([next_timestamp], [pred_value], color='red', s=100, 
                      label=f'Predicted {metric_name}: {pred_value:.2f}')
            
            # Add vertical line at current time
            ax.axvline(x=df[timestamp_col].max(), color='black', linestyle='--', alpha=0.5, 
                      label='Current Time')
            
            # Set reasonable limits based on recent data
            y_min = min(last_24h[col_name].min() * 0.9, pred_value * 0.9)
            y_max = max(last_24h[col_name].max() * 1.1, pred_value * 1.1)
            ax.set_ylim(y_min, y_max)
            
            # Zoom in to the relevant time range
            ax.set_xlim(last_24h[timestamp_col].min(), next_timestamp + pd.Timedelta(minutes=15))
        
        # Add labels and grid
        ax.set_title(f'{metric_name} Prediction for Sensor {sensor_id}')
        ax.set_xlabel('Time')
        ax.set_ylabel(metric_name)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path)
        print(f"Prediction visualization saved to {save_path}")
    
    return fig 