import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import os
from joblib import Memory

# Thiết lập thư mục cache cho data_transform
cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'cache'))
os.makedirs(cache_dir, exist_ok=True)
memory = Memory(location=cache_dir, verbose=0)

def transform_df(df: pd.DataFrame) -> pd.DataFrame:
    # Drop specified columns
    columns_to_drop = ['_id', 'count', 'sum', 'average', 'min', 'max',
                      'startDate', 'endDate', 'lastDataUpdateTime', 'dataType']
    df = df.drop(columns=columns_to_drop, errors='ignore')


    # Define column mapping based on chNumber
    ch_map = {0: '_Pressure_1', 1: '_Flow', 2: '_Pressure_2'}

    # Generate column names for timestamps and values
    time_cols = np.array([f'dataValues.{i}.dataTime' for i in range(96)])
    value_cols = np.array([f'dataValues.{i}.dataValue' for i in range(96)])

    # Filter valid chNumber values
    df = df[df['chNumber'].isin(ch_map.keys())].copy()

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

    # Filter out invalid rows (NaN timestamps or values)
    # long_df = long_df[long_df['Timestamp'].notna() & long_df['Value'].notna()]
    long_df = long_df[long_df['Timestamp'].notna()]

    # Pivot the data efficiently
    result_df = long_df.pivot_table(
        index='Timestamp',
        columns='Column',
        values='Value',
        aggfunc='first'  # Use 'mean' or 'last' if preferred
    ).reset_index()

    # Sort by timestamp
    result_df = result_df.sort_values('Timestamp').reset_index(drop=True)

    return result_df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    result_df = df.copy()
    
    sensor_ids = []
    for col in df.columns:
        if col.endswith('_Pressure_2'):
            sensor_id = col.split('_')[0]
            if (f'{sensor_id}_Pressure_1' in df.columns) and (f'{sensor_id}_Flow' in df.columns):
                sensor_ids.append(sensor_id)
    
    for sensor_id in sensor_ids:
        p1_col = f'{sensor_id}_Pressure_1'
        p2_col = f'{sensor_id}_Pressure_2'
        diff_col = f'{sensor_id}_Pressure_Diff'
        
        valid_mask = df[p1_col].notna() & df[p2_col].notna()
        if valid_mask.any():
            result_df[diff_col] = df[p2_col] - df[p1_col]
    
    return result_df


def identify_potential_leaks(df: pd.DataFrame, 
                            threshold_diff: float = 0.5, 
                            window_size: int = 10) -> Dict[str, pd.DataFrame]:
    if df is None or df.empty:
        return {}
    
    result = {}
    
    diff_cols = [col for col in df.columns if col.endswith('_Pressure_Diff')]
    
    for diff_col in diff_cols:
        sensor_id = diff_col.split('_')[0]
        
        if df[diff_col].isna().all():
            continue
            
        rolling_mean = df[diff_col].rolling(window=window_size).mean()
        rolling_std = df[diff_col].rolling(window=window_size).std()
        
        is_anomaly = abs(df[diff_col] - rolling_mean) > (threshold_diff * rolling_std)
        
        if is_anomaly.any():
            leak_events = df[is_anomaly].copy()
            if not leak_events.empty:
                result[sensor_id] = leak_events
    
    return result