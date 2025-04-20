"""
Data loading functionality for leak detection system.
"""

import pandas as pd
import os
from typing import Dict, List, Optional, Union


class DataLoader:
    """Class for loading and accessing water leakage data files."""
    
    def __init__(self, base_path: str):
        """
        Initialize DataLoader with the base path to data files.

        Args:
            base_path (str): Base directory containing the data files
        
        Raises:
            ValueError: If the base_path does not exist
        """
        if not os.path.exists(base_path):
            raise ValueError(f"Data directory not found: {base_path}")
            
        self.base_path = base_path
        self.file_paths = {
            'query_1': os.path.join(base_path, 'find_query_1.csv'),
            'query_2': os.path.join(base_path, 'find_query_2.csv'),
            'query_3': os.path.join(base_path, 'find_query_3.csv'),
            'channel_type': os.path.join(base_path, 'channel_data_type.csv')
        }
        
        # Validate file existence
        for name, path in self.file_paths.items():
            if not os.path.exists(path):
                print(f"Warning: {name} file not found at path: {path}")

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV files and return as a dictionary of DataFrames.

        Returns:
            dict: Dictionary containing DataFrames for each data file
            
        Raises:
            FileNotFoundError: If any required data files are missing
        """
        data = {}
        
        # Load each file with error handling
        for name, path in self.file_paths.items():
            try:
                data[name] = pd.read_csv(path)
            except FileNotFoundError:
                raise FileNotFoundError(f"Required data file not found: {path}")
            except pd.errors.ParserError:
                raise ValueError(f"Error parsing CSV file: {path}")
        
        # Create a merged dataset
        data['merged_data'] = pd.concat(
            [data['query_1'], data['query_2'], data['query_3']], 
            ignore_index=True
        )
        
        return data
    
    def get_sensor_ids(self, result_df: pd.DataFrame) -> List[str]:
        """
        Extract sensor IDs from the transformed dataframe.
        
        Args:
            result_df (pd.DataFrame): Transformed DataFrame with sensor columns
            
        Returns:
            list: List of sensor IDs
        """
        if result_df is None or result_df.empty:
            return []
            
        # Find columns ending with '_Pressure_2' to extract sensor IDs
        filtered_columns = result_df.columns[result_df.columns.str.endswith('_Pressure_2')]
        sensor_ids = [col.split('_')[0] for col in filtered_columns]
        return sensor_ids
    
    def filter_sensors_with_complete_data(self, result_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the dataframe to include only sensors with all three metrics.
        
        Args:
            result_df (pd.DataFrame): Transformed DataFrame with sensor columns
            
        Returns:
            pd.DataFrame: Filtered DataFrame with only complete sensor data
        """
        if result_df is None or result_df.empty:
            return pd.DataFrame()
            
        sensor_ids = self.get_sensor_ids(result_df)
        
        cols = ['Timestamp']
        valid_sensor_count = 0
        
        for sensor_id in sensor_ids:
            flow_col = f'{sensor_id}_Flow'
            pressure_1_col = f'{sensor_id}_Pressure_1'
            pressure_2_col = f'{sensor_id}_Pressure_2'
            
            # Only include sensors that have all three metrics
            if all(col in result_df.columns for col in [flow_col, pressure_1_col, pressure_2_col]):
                cols.extend([flow_col, pressure_1_col, pressure_2_col])
                valid_sensor_count += 1
        
        if valid_sensor_count == 0:
            print("Warning: No sensors with complete data (flow, pressure_1, pressure_2) found.")
            
        # Select only the columns we need
        new_df = result_df[cols].copy() if cols else pd.DataFrame()
        return new_df
        
    def get_data_file_info(self) -> Dict[str, dict]:
        """
        Get information about the data files.
        
        Returns:
            dict: Dictionary with file information (exists, size)
        """
        file_info = {}
        
        for name, path in self.file_paths.items():
            info = {
                'exists': os.path.exists(path),
                'size': os.path.getsize(path) if os.path.exists(path) else 0
            }
            file_info[name] = info
            
        return file_info 