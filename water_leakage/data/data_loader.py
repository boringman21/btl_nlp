"""
Data loading functionality for leak detection system.
"""

import pandas as pd
import os

class DataLoader:
    def __init__(self, base_path):
        """
        Initialize DataLoader with the base path to data files.

        Args:
            base_path (str): Base directory containing the data files
        """
        self.base_path = base_path
        self.query_1_path = os.path.join(base_path, 'find_query_1.csv')
        self.query_2_path = os.path.join(base_path, 'find_query_2.csv')
        self.query_3_path = os.path.join(base_path, 'find_query_3.csv')
        self.channel_type_path = os.path.join(base_path, 'channel_data_type.csv')

    def load_all_data(self):
        """
        Load all CSV files and return as a dictionary of DataFrames.

        Returns:
            dict: Dictionary containing DataFrames for each data file
        """
        data = {
            'query_1': pd.read_csv(self.query_1_path),
            'query_2': pd.read_csv(self.query_2_path),
            'query_3': pd.read_csv(self.query_3_path),
            'channel_type': pd.read_csv(self.channel_type_path)
        }
        
        # Create a merged dataset
        data['merged_data'] = pd.concat(
            [data['query_1'], data['query_2'], data['query_3']], 
            ignore_index=True
        )
        
        return data
    
    def get_sensor_ids(self, result_df):
        """
        Extract sensor IDs from the transformed dataframe.
        
        Args:
            result_df (pd.DataFrame): Transformed DataFrame with sensor columns
            
        Returns:
            list: List of sensor IDs
        """
        filtered_columns = result_df.columns[result_df.columns.str.endswith('_Pressure_2')]
        sensor_ids = [col.split('_')[0] for col in filtered_columns]
        return sensor_ids
    
    def filter_sensors_with_complete_data(self, result_df):
        """
        Filter the dataframe to include only sensors with all three metrics.
        
        Args:
            result_df (pd.DataFrame): Transformed DataFrame with sensor columns
            
        Returns:
            pd.DataFrame: Filtered DataFrame with only complete sensor data
        """
        sensor_ids = self.get_sensor_ids(result_df)
        
        cols = ['Timestamp']
        
        for sensor_id in sensor_ids:
            flow_col = f'{sensor_id}_Flow'
            pressure_1_col = f'{sensor_id}_Pressure_1'
            pressure_2_col = f'{sensor_id}_Pressure_2'
            
            # Only include sensors that have all three metrics
            if all(col in result_df.columns for col in [flow_col, pressure_1_col, pressure_2_col]):
                cols.append(flow_col)
                cols.append(pressure_1_col)
                cols.append(pressure_2_col)
        
        new_df = result_df[cols]
        return new_df 