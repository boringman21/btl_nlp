#!/usr/bin/env python3
"""
Visualization utilities for water leakage data.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def plot_single_metric(df, sensor_id, metric, save_dir=None):
    """
    Plot a single metric for a given sensor.
    
    Args:
        df (pd.DataFrame): DataFrame with sensor data
        sensor_id (str): Sensor ID
        metric (str): Metric to plot (Flow, Pressure_1, Pressure_2, Pressure_Diff)
        save_dir (str, optional): Directory to save the plot
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    sensor_id = str(sensor_id)
    
    # Check if column exists
    col = f"{sensor_id}_{metric}"
    
    if col not in df.columns:
        print(f"No {metric} data available for sensor {sensor_id}")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot metric
    ax.plot(df['Timestamp'], df[col], label=metric, color='blue')
    
    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel(f'{metric} Value')
    ax.set_title(f'{metric} for Sensor {sensor_id}')
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Save figure if save_dir is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{metric}.png")
        plt.savefig(save_path)
        print(f"{metric} plot saved to: {save_path}")
    
    return fig

def plot_all_metrics(df, sensor_id, save_dir=None):
    """
    Plot all available metrics (flow, pressure_1, pressure_2, pressure_diff) for a given sensor.
    
    Args:
        df (pd.DataFrame): DataFrame with sensor data
        sensor_id (str): Sensor ID
        save_dir (str, optional): Directory to save the plot
        
    Returns:
        matplotlib.figure.Figure: Figure object
        bool: Whether sensor has two pressure metrics or only one
    """
    sensor_id = str(sensor_id)
    
    # Check if required columns exist
    flow_col = f"{sensor_id}_Flow"
    pressure_1_col = f"{sensor_id}_Pressure_1"
    pressure_2_col = f"{sensor_id}_Pressure_2"
    pressure_diff_col = f"{sensor_id}_Pressure_Diff"
    
    columns_to_plot = []
    if flow_col in df.columns:
        columns_to_plot.append((flow_col, 'Flow', 'blue'))
    if pressure_1_col in df.columns:
        columns_to_plot.append((pressure_1_col, 'Pressure 1', 'green'))
    if pressure_2_col in df.columns:
        columns_to_plot.append((pressure_2_col, 'Pressure 2', 'red'))
    if pressure_diff_col in df.columns:
        columns_to_plot.append((pressure_diff_col, 'Pressure Difference', 'purple'))
    
    has_two_pressure = pressure_1_col in df.columns and pressure_2_col in df.columns
    
    if not columns_to_plot:
        print(f"No data available for sensor {sensor_id}")
        return None, False
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each column
    for col, label, color in columns_to_plot:
        ax.plot(df['Timestamp'], df[col], label=label, color=color)
    
    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(f'All Metrics for Sensor {sensor_id}')
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Save figure if save_dir is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"all_metrics.png")
        plt.savefig(save_path)
        print(f"All metrics plot saved to: {save_path}")
    
    return fig, has_two_pressure

def plot_flow_and_pressure(df, sensor_id, save_dir=None):
    """
    Plot flow and pressure data for a given sensor.
    
    Args:
        df (pd.DataFrame): DataFrame with sensor data
        sensor_id (str): Sensor ID
        save_dir (str, optional): Directory to save the plot
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    sensor_id = str(sensor_id)
    
    # Check if required columns exist
    flow_col = f"{sensor_id}_Flow"
    pressure_1_col = f"{sensor_id}_Pressure_1"
    pressure_2_col = f"{sensor_id}_Pressure_2"
    pressure_diff_col = f"{sensor_id}_Pressure_Diff"
    
    columns_to_plot = []
    if flow_col in df.columns:
        columns_to_plot.append((flow_col, 'Flow', 'blue'))
    if pressure_1_col in df.columns:
        columns_to_plot.append((pressure_1_col, 'Pressure 1', 'green'))
    if pressure_2_col in df.columns:
        columns_to_plot.append((pressure_2_col, 'Pressure 2', 'red'))
    if pressure_diff_col in df.columns:
        columns_to_plot.append((pressure_diff_col, 'Pressure Difference', 'purple'))
    
    if not columns_to_plot:
        print(f"No data available for sensor {sensor_id}")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each column
    for col, label, color in columns_to_plot:
        ax.plot(df['Timestamp'], df[col], label=label, color=color)
    
    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(f'Flow and Pressure for Sensor {sensor_id}')
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Save figure if save_dir is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{sensor_id}_flow_pressure.png")
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    
    return fig

def plot_correlation(df, sensor_id, save_dir=None):
    """
    Plot correlation matrix for a given sensor.
    
    Args:
        df (pd.DataFrame): DataFrame with sensor data
        sensor_id (str): Sensor ID
        save_dir (str, optional): Directory to save the plot
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    sensor_id = str(sensor_id)
    
    # Get columns for this sensor
    sensor_cols = [col for col in df.columns if col.startswith(f"{sensor_id}_")]
    
    if not sensor_cols:
        print(f"No data available for sensor {sensor_id}")
        return None
    
    # Create correlation matrix
    corr_df = df[sensor_cols].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', ax=ax)
    
    # Add title
    ax.set_title(f'Correlation Matrix for Sensor {sensor_id}')
    
    # Save figure if save_dir is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"correlation.png")
        plt.savefig(save_path)
        print(f"Correlation matrix saved to: {save_path}")
    
    return fig

def plot_all_sensors(df, metric='Flow', save_dir=None):
    """
    Plot same metric for all sensors.
    
    Args:
        df (pd.DataFrame): DataFrame with sensor data
        metric (str): Metric to plot (Flow, Pressure_1, Pressure_2, Pressure_Diff)
        save_dir (str, optional): Directory to save the plot
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Find all columns with the specified metric
    metric_cols = [col for col in df.columns if col.endswith(f"_{metric}")]
    
    if not metric_cols:
        print(f"No {metric} data available for any sensor")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each sensor
    for col in metric_cols:
        sensor_id = col.split('_')[0]
        ax.plot(df['Timestamp'], df[col], label=f"Sensor {sensor_id}")
    
    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel(f'{metric} Value')
    ax.set_title(f'{metric} for All Sensors')
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Save figure if save_dir is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"all_sensors_{metric}.png")
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    
    return fig

def plot_metrics_in_subplots(df, sensor_id, save_dir=None):
    """
    Plot all available metrics (flow, pressure_1, pressure_2, pressure_diff) for a given sensor
    in separate subplots of a single figure.
    
    Args:
        df (pd.DataFrame): DataFrame with sensor data
        sensor_id (str): Sensor ID
        save_dir (str, optional): Directory to save the plot
        
    Returns:
        matplotlib.figure.Figure: Figure object
        bool: Whether sensor has two pressure metrics or only one
    """
    sensor_id = str(sensor_id)
    
    # Check if required columns exist
    flow_col = f"{sensor_id}_Flow"
    pressure_1_col = f"{sensor_id}_Pressure_1"
    pressure_2_col = f"{sensor_id}_Pressure_2"
    pressure_diff_col = f"{sensor_id}_Pressure_Diff"
    
    # Identify which metrics are available
    available_metrics = []
    if flow_col in df.columns:
        available_metrics.append(('Flow', flow_col, 'blue'))
    if pressure_1_col in df.columns:
        available_metrics.append(('Pressure_1', pressure_1_col, 'green'))
    if pressure_2_col in df.columns:
        available_metrics.append(('Pressure_2', pressure_2_col, 'red'))
    if pressure_diff_col in df.columns:
        available_metrics.append(('Pressure_Diff', pressure_diff_col, 'purple'))
    
    has_two_pressure = pressure_1_col in df.columns and pressure_2_col in df.columns
    
    if not available_metrics:
        print(f"No data available for sensor {sensor_id}")
        return None, False
    
    # Create figure with subplots - one for each metric
    num_metrics = len(available_metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 3*num_metrics), sharex=True)
    
    # If only one metric, axes will not be an array
    if num_metrics == 1:
        axes = [axes]
    
    # Plot each metric in a separate subplot
    for i, (metric_name, col, color) in enumerate(available_metrics):
        axes[i].plot(df['Timestamp'], df[col], color=color)
        
        # Add labels and title for each subplot
        axes[i].set_ylabel(metric_name)
        axes[i].set_title(f'{metric_name} for Sensor {sensor_id}')
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    # Add common x-label
    axes[-1].set_xlabel('Time')
    
    # Format dates on x-axis if needed
    if pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        fig.autofmt_xdate()
    
    # Add overall title
    plt.suptitle(f'All Metrics for Sensor {sensor_id}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for the suptitle
    
    # Save figure if save_dir is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"metrics_combined.png")
        plt.savefig(save_path)
        print(f"Combined metrics plot saved to: {save_path}")
    
    return fig, has_two_pressure

def create_sensor_visualizations(df, sensor_ids, output_dir, separate_metrics=False):
    """
    Create visualizations for each sensor with the requested directory structure.
    
    Args:
        df (pd.DataFrame): DataFrame with sensor data
        sensor_ids (list): List of sensor IDs to visualize
        output_dir (str): Base output directory
        separate_metrics (bool): If True, plot each metric separately
    """
    # Create main visualization directory
    viz_dir = os.path.join(output_dir, "visualization")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create directories for different pressure types
    one_pressure_dir = os.path.join(viz_dir, "1_pressure")
    two_pressure_dir = os.path.join(viz_dir, "2_pressure")
    os.makedirs(one_pressure_dir, exist_ok=True)
    os.makedirs(two_pressure_dir, exist_ok=True)
    
    # Process each sensor
    for sensor_id in sensor_ids:
        # Check if this sensor has two pressure sensors or just one
        pressure_1_col = f"{sensor_id}_Pressure_1"
        pressure_2_col = f"{sensor_id}_Pressure_2"
        
        has_two_pressure = pressure_1_col in df.columns and pressure_2_col in df.columns
        
        # Create sensor directory in the appropriate parent directory
        if has_two_pressure:
            sensor_dir = os.path.join(two_pressure_dir, sensor_id)
        else:
            sensor_dir = os.path.join(one_pressure_dir, sensor_id)
        
        os.makedirs(sensor_dir, exist_ok=True)
        
        if separate_metrics:
            # Create separate plots for each metric
            metrics = ['Flow', 'Pressure_1', 'Pressure_2', 'Pressure_Diff']
            for metric in metrics:
                fig = plot_single_metric(df, sensor_id, metric, save_dir=sensor_dir)
                plt.close(fig) if fig else None
                
            # Also create combined plot with subplots
            fig, _ = plot_metrics_in_subplots(df, sensor_id, save_dir=sensor_dir)
            plt.close(fig) if fig else None
        else:
            # Create all metrics visualization
            fig, _ = plot_all_metrics(df, sensor_id, save_dir=sensor_dir)
            plt.close(fig) if fig else None
        
        # Create correlation matrix
        fig = plot_correlation(df, sensor_id, save_dir=sensor_dir)
        plt.close(fig) if fig else None
        
    print(f"Sensor visualizations created in {viz_dir}")

def plot_predictions(df, prediction, sensor_id, next_timestamp, save_path=None):
    """
    Plot historical data with prediction for the next tick.
    
    Args:
        df (pd.DataFrame): DataFrame with historical data
        prediction (dict): Dictionary with predicted values
        sensor_id (str): Sensor ID
        next_timestamp (datetime): Timestamp for the prediction
        save_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Convert timestamp column to datetime if needed
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Get the last 24 hours of data
    last_timestamp = df['Timestamp'].max()
    start_time = last_timestamp - timedelta(hours=24)
    recent_df = df[df['Timestamp'] >= start_time].copy()
    
    # Create a figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    metrics = ['Flow', 'Pressure_1', 'Pressure_2']
    colors = ['#3388ff', '#ff5555', '#55aa55']
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        col = f"{sensor_id}_{metric}"
        if col in df.columns:
            # Plot historical data
            axes[i].plot(recent_df['Timestamp'], recent_df[col], 
                        label=f'Historical {metric}', color=colors[i])
            
            # Add the prediction point
            if metric in prediction:
                pred_value = prediction[metric]
                axes[i].scatter([next_timestamp], [pred_value], 
                              label=f'Predicted {metric}', 
                              color='red', s=100, zorder=5)
                
                # Add text label
                axes[i].text(next_timestamp, pred_value, 
                           f'{pred_value:.2f}', 
                           color='red', fontweight='bold',
                           ha='left', va='bottom')
            
            axes[i].set_title(f'{metric} for Sensor {sensor_id}')
            axes[i].set_ylabel(metric)
            axes[i].legend(loc='upper left')
            axes[i].grid(True, linestyle='--', alpha=0.7)
    
    # Format x-axis
    axes[-1].set_xlabel('Time')
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Add vertical line for current time
    for ax in axes:
        ax.axvline(x=last_timestamp, color='gray', linestyle='--', alpha=0.7)
        ax.text(last_timestamp, ax.get_ylim()[1]*0.9, 'Current', 
               color='gray', rotation=90, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Prediction plot saved to: {save_path}")
    
    return fig
