"""
Data transformation functionality for leak detection system.
"""

import pandas as pd
import numpy as np

def transform_df(df):
    """
    Transform raw data into a time-series format with sensors as columns.
    
    Args:
        df (pd.DataFrame): Raw DataFrame from the merged queries
        
    Returns:
        pd.DataFrame: Transformed DataFrame with timestamps as index and sensor data as columns
    """
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

    # Filter out invalid rows (NaN timestamps)
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
    
    # Convert Timestamp to datetime
    result_df['Timestamp'] = pd.to_datetime(result_df['Timestamp'])

    return result_df

def add_derived_metrics(df):
    """
    Add derived metrics to the transformed dataframe.
    
    Args:
        df (pd.DataFrame): Transformed DataFrame with sensor data
        
    Returns:
        pd.DataFrame: DataFrame with added derived metrics
    """
    # Get list of sensor IDs
    sensor_ids = []
    for col in df.columns:
        if col.endswith('_Pressure_2'):
            sensor_id = col.split('_')[0]
            if (f'{sensor_id}_Pressure_1' in df.columns) and (f'{sensor_id}_Flow' in df.columns):
                sensor_ids.append(sensor_id)
    
    # Add pressure difference for each sensor
    for sensor_id in sensor_ids:
        p1_col = f'{sensor_id}_Pressure_1'
        p2_col = f'{sensor_id}_Pressure_2'
        diff_col = f'{sensor_id}_Pressure_Diff'
        
        df[diff_col] = df[p2_col] - df[p1_col]
    
    return df 