"""
Data visualization utilities for the leak detection system.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_flow_and_pressure(result_df, sensor_id, save_dir=None):
    """
    Plots flow and pressure data from the given DataFrame for the specified sensor ID.
    Optionally saves the plots to the specified directory.
    
    Args:
        result_df (pd.DataFrame): DataFrame containing sensor data
        sensor_id (str or int): The sensor ID to plot
        save_dir (str, optional): Directory to save the plots. If None, plots are only displayed.
    
    Returns:
        plt.Figure: The matplotlib figure object
    """
    sensor_id = str(sensor_id)
    flow_col = f'{sensor_id}_Flow'
    pressure_1_col = f'{sensor_id}_Pressure_1'
    pressure_2_col = f'{sensor_id}_Pressure_2'
    timestamp_col = 'Timestamp'

    # Create directory if save_dir is specified
    if save_dir is not None:
        folder_path = os.path.join(save_dir, sensor_id)
        os.makedirs(folder_path, exist_ok=True)
    else:
        folder_path = None

    # Check if the timestamp column exists
    if timestamp_col not in result_df.columns:
        print(f"Error: Missing '{timestamp_col}' column in DataFrame.")
        return None

    # Convert Timestamp to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(result_df[timestamp_col]):
        result_df[timestamp_col] = pd.to_datetime(result_df[timestamp_col])

    # Initialize the plot
    fig = plt.figure(figsize=(12, 10))

    plot_count = 1

    # Plot Flow if available
    if flow_col in result_df.columns:
        plt.subplot(4, 1, plot_count)
        plt.plot(result_df[timestamp_col], result_df[flow_col], label='Flow', color='blue')
        plt.title(f'Flow vs Time (Sensor {sensor_id})')
        plt.xlabel('Timestamp')
        plt.ylabel('Flow')
        plt.grid(True)
        plt.legend()
        plot_count += 1

    # Plot Pressure_1 if available
    if pressure_1_col in result_df.columns:
        plt.subplot(4, 1, plot_count)
        plt.plot(result_df[timestamp_col], result_df[pressure_1_col], label='Pressure_1', color='green')
        plt.title(f'Pressure_1 vs Time (Sensor {sensor_id})')
        plt.xlabel('Timestamp')
        plt.ylabel('Pressure_1')
        plt.grid(True)
        plt.legend()
        plot_count += 1

    # Plot Pressure_2 if available
    if pressure_2_col in result_df.columns:
        plt.subplot(4, 1, plot_count)
        plt.plot(result_df[timestamp_col], result_df[pressure_2_col], label='Pressure_2', color='red')
        plt.title(f'Pressure_2 vs Time (Sensor {sensor_id})')
        plt.xlabel('Timestamp')
        plt.ylabel('Pressure_2')
        plt.grid(True)
        plt.legend()
        plot_count += 1

    # Plot Pressure Difference if both Pressure_1 and Pressure_2 are available
    if pressure_1_col in result_df.columns and pressure_2_col in result_df.columns:
        pressure_diff = result_df[pressure_2_col] - result_df[pressure_1_col]
        plt.subplot(4, 1, plot_count)
        plt.plot(result_df[timestamp_col], pressure_diff, label='Pressure_2 - Pressure_1', color='purple')
        plt.title(f'Pressure Difference (Pressure_2 - Pressure_1) vs Time (Sensor {sensor_id})')
        plt.xlabel('Timestamp')
        plt.ylabel('Pressure Difference')
        plt.grid(True)
        plt.legend()
        plot_count += 1

    # Adjust layout and save if folder_path is specified
    if plot_count > 1:
        plt.tight_layout()
        if folder_path:
            save_path = os.path.join(folder_path, f"sensor_{sensor_id}_plot.png")
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
    else:
        print("No valid data to plot.")
        plt.close(fig)
        return None

    return fig

def plot_correlation(result_df, sensor_id, save_dir=None):
    """
    Plots the correlation matrix of flow and pressure data for the specified sensor ID.
    
    Args:
        result_df (pd.DataFrame): DataFrame containing sensor data
        sensor_id (str or int): The sensor ID to analyze
        save_dir (str, optional): Directory to save the plot. If None, plot is only displayed.
    
    Returns:
        plt.Figure: The matplotlib figure object
    """
    sensor_id = str(sensor_id)
    flow_col = f'{sensor_id}_Flow'
    pressure_1_col = f'{sensor_id}_Pressure_1'
    pressure_2_col = f'{sensor_id}_Pressure_2'

    # Create directory if save_dir is specified
    if save_dir is not None:
        folder_path = os.path.join(save_dir, sensor_id)
        os.makedirs(folder_path, exist_ok=True)
    else:
        folder_path = None

    # Check if required columns exist
    required_cols = [flow_col, pressure_1_col, pressure_2_col]
    missing_cols = [col for col in required_cols if col not in result_df.columns]
    if missing_cols:
        print(f"Error: Missing columns in DataFrame: {', '.join(missing_cols)}")
        return None

    # Extract relevant columns
    df = result_df[[flow_col, pressure_1_col, pressure_2_col]]

    # Calculate correlation matrix
    corr_matrix = df.corr()

    # Plot the correlation matrix as a heatmap
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title(f'Correlation Matrix (Sensor {sensor_id})')

    # Save the plot if folder_path is specified
    if folder_path:
        save_path = os.path.join(folder_path, f"sensor_{sensor_id}_correlation.png")
        plt.savefig(save_path)
        print(f"Correlation plot saved to {save_path}")

    return fig 