import pandas as pd
import os
from typing import Dict, List, Optional, Union


class DataLoader:
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.file_paths = {
            'query_1': os.path.join(base_path, 'find_query_1.csv'),
            'query_2': os.path.join(base_path, 'find_query_2.csv'),
            'query_3': os.path.join(base_path, 'find_query_3.csv'),
            'channel_type': os.path.join(base_path, 'channel_data_type.csv')
        }

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        data = {}
        
        for name, path in self.file_paths.items():
            data[name] = pd.read_csv(path)
        
        data['merged_data'] = pd.concat(
            [data['query_1'], data['query_2'], data['query_3']], 
            ignore_index=True
        )
        
        return data
    
    def get_sensor_ids(self, result_df: pd.DataFrame) -> List[str]:
        if result_df is None or result_df.empty:
            return []
            
        filtered_columns = result_df.columns[result_df.columns.str.endswith('_Pressure_2')]
        sensor_ids = [col.split('_')[0] for col in filtered_columns]
        return sensor_ids
    
    def filter_sensors_with_complete_data(self, result_df: pd.DataFrame) -> pd.DataFrame:
        if result_df is None or result_df.empty:
            return pd.DataFrame()
            
        sensor_ids = self.get_sensor_ids(result_df)
        
        cols = ['Timestamp']
        
        for sensor_id in sensor_ids:
            flow_col = f'{sensor_id}_Flow'
            pressure_1_col = f'{sensor_id}_Pressure_1'
            pressure_2_col = f'{sensor_id}_Pressure_2'
            
            if all(col in result_df.columns for col in [flow_col, pressure_1_col, pressure_2_col]):
                cols.extend([flow_col, pressure_1_col, pressure_2_col])
                
        new_df = result_df[cols].copy() if cols else pd.DataFrame()
        return new_df
        
    def get_data_file_info(self) -> Dict[str, dict]:
        file_info = {}
        
        for name, path in self.file_paths.items():
            info = {
                'exists': os.path.exists(path),
                'size': os.path.getsize(path) if os.path.exists(path) else 0
            }
            file_info[name] = info
            
        return file_info 


def load_data(data_path: str) -> Optional[pd.DataFrame]:
    loader = DataLoader(data_path)
    data_dict = loader.load_all_data()
    return data_dict['merged_data']