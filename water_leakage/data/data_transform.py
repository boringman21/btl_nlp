"""
Data transformation functionality for leak detection system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple


def transform_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw data into a time-series format with sensors as columns.
    
    Args:
        df (pd.DataFrame): Raw DataFrame from the merged queries
        
    Returns:
        pd.DataFrame: Transformed DataFrame with timestamps as index and sensor data as columns
        
    Raises:
        ValueError: If df is None or empty
        KeyError: If required columns are missing
    """
    # Validate input
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None")
    
    required_cols = ['smsNumber', 'chNumber']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing from input DataFrame")
    
    # Drop specified columns
    columns_to_drop = ['_id', 'count', 'sum', 'average', 'min', 'max',
                      'startDate', 'endDate', 'lastDataUpdateTime', 'dataType']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Define column mapping based on chNumber
    ch_map = {0: '_Pressure_1', 1: '_Flow', 2: '_Pressure_2'}

    # Generate column names for timestamps and values
    time_cols = np.array([f'dataValues.{i}.dataTime' for i in range(96)])
    value_cols = np.array([f'dataValues.{i}.dataValue' for i in range(96)])

    # Check if time and value columns exist
    missing_time_cols = [col for col in time_cols if col not in df.columns]
    missing_value_cols = [col for col in value_cols if col not in df.columns]
    if missing_time_cols or missing_value_cols:
        print(f"Warning: {len(missing_time_cols)} time columns and {len(missing_value_cols)} value columns are missing")

    # Filter valid chNumber values
    df = df[df['chNumber'].isin(ch_map.keys())].copy()
    if df.empty:
        raise ValueError("No valid chNumber values found in DataFrame")

    # Create column names using vectorized operations
    df['col_name'] = df['smsNumber'].astype(str) + df['chNumber'].map(ch_map)

    # Convert DataFrame to NumPy arrays for faster processing
    timestamps = df[time_cols].to_numpy()  # Shape: (n_rows, 96)
    values = df[value_cols].to_numpy()     # Shape: (n_rows, 96)
    col_names = df['col_name'].to_numpy()  # Shape: (n_rows,)

    # Reshape into long format using NumPy
    n_rows = len(df)
    timestamps_flat = timestamps.reshape(n_rows * 96)  # Shape: (n_rows * 96,)
    values_flat = values.reshape(n_rows * 96)          # Shape: (n_rows * 96,)
    col_names_flat = np.repeat(col_names, 96)          # Shape: (n_rows * 96,)

    # Create a single DataFrame with all data
    long_df = pd.DataFrame({
        'Timestamp': timestamps_flat,
        'Value': values_flat,
        'Column': col_names_flat
    })

    # Filter out invalid rows (NaN timestamps)
    long_df = long_df[long_df['Timestamp'].notna()]
    if long_df.empty:
        raise ValueError("No valid timestamp data after filtering")

    # Pivot the data efficiently
    result_df = long_df.pivot_table(
        index='Timestamp',
        columns='Column',
        values='Value',
        aggfunc='first'  # Use 'mean' or 'last' if preferred
    ).reset_index()

    # Sort by timestamp
    result_df = result_df.sort_values('Timestamp').reset_index(drop=True)
    
    # Convert Timestamp to datetime
    try:
        result_df['Timestamp'] = pd.to_datetime(result_df['Timestamp'])
    except Exception as e:
        print(f"Warning: Could not convert timestamps to datetime format: {str(e)}")

    return result_df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived metrics to the transformed dataframe.
    
    Args:
        df (pd.DataFrame): Transformed DataFrame with sensor data
        
    Returns:
        pd.DataFrame: DataFrame with added derived metrics
        
    Raises:
        ValueError: If df is None or empty
    """
    # Validate input
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None")
    
    if 'Timestamp' not in df.columns:
        raise KeyError("Required 'Timestamp' column missing from DataFrame")
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Get list of sensor IDs
    sensor_ids = []
    for col in df.columns:
        if col.endswith('_Pressure_2'):
            sensor_id = col.split('_')[0]
            if (f'{sensor_id}_Pressure_1' in df.columns) and (f'{sensor_id}_Flow' in df.columns):
                sensor_ids.append(sensor_id)
    
    if not sensor_ids:
        print("Warning: No complete sensor data found (sensors with all metrics)")
        return result_df
    
    # Add pressure difference for each sensor
    for sensor_id in sensor_ids:
        p1_col = f'{sensor_id}_Pressure_1'
        p2_col = f'{sensor_id}_Pressure_2'
        diff_col = f'{sensor_id}_Pressure_Diff'
        
        # Check for NaN values before calculating difference
        valid_mask = df[p1_col].notna() & df[p2_col].notna()
        if valid_mask.any():
            result_df[diff_col] = df[p2_col] - df[p1_col]
        else:
            print(f"Warning: Sensor {sensor_id} has no valid pressure data for calculating difference")
    
    return result_df


def identify_potential_leaks(df: pd.DataFrame, 
                            threshold_diff: float = 0.5, 
                            window_size: int = 10) -> Dict[str, pd.DataFrame]:
    """
    Identify potential leaks by looking for sudden changes in pressure difference.
    
    Args:
        df (pd.DataFrame): DataFrame with sensor data including pressure differences
        threshold_diff (float): Threshold for pressure difference change
        window_size (int): Window size for rolling calculations
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of potential leak events by sensor
    """
    if df is None or df.empty:
        return {}
    
    result = {}
    
    # Find all pressure difference columns
    diff_cols = [col for col in df.columns if col.endswith('_Pressure_Diff')]
    
    for diff_col in diff_cols:
        sensor_id = diff_col.split('_')[0]
        
        # Skip if no valid data
        if df[diff_col].isna().all():
            continue
            
        # Calculate rolling mean and standard deviation
        rolling_mean = df[diff_col].rolling(window=window_size).mean()
        rolling_std = df[diff_col].rolling(window=window_size).std()
        
        # Find significant deviations
        is_anomaly = abs(df[diff_col] - rolling_mean) > (threshold_diff * rolling_std)
        
        # Store potential leak events
        if is_anomaly.any():
            leak_events = df[is_anomaly].copy()
            if not leak_events.empty:
                result[sensor_id] = leak_events
    
    return result 