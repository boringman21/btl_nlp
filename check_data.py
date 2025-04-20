#!/usr/bin/env python3
"""
Script to verify data transformation from CSV to DataFrame.
"""

from water_leakage.data.data_loader import DataLoader
from water_leakage.data.data_transform import transform_df, add_derived_metrics
import pandas as pd
import numpy as np

def main():
    print("=== Checking Data Transformation ===")
    
    # 1. Load data from CSV
    print("\n1. Loading data from CSV files...")
    loader = DataLoader('./LeakDataset/Logger_Data_2024_Bau_Bang-2')
    data = loader.load_all_data()
    
    # 2. Check loaded data
    print("\n2. CSV files loaded:")
    for key, df in data.items():
        print(f"- {key}: {df.shape}")
    
    # 3. Check original data structure
    print("\n3. Original data structure (merged_data):")
    original_df = data['merged_data']
    print(f"Columns: {original_df.columns.tolist()[:10]}...")
    print(f"Data types: {original_df.dtypes[:5]}")
    
    # 4. Check for missing columns before transformation
    required_cols = ['smsNumber', 'chNumber']
    missing_cols = [col for col in required_cols if col not in original_df.columns]
    print(f"\n4. Missing required columns: {missing_cols if missing_cols else 'None'}")
    
    # 5. Sample of data values column names
    data_value_cols = [col for col in original_df.columns if col.startswith('dataValues')]
    print(f"\n5. Sample of data value columns: {data_value_cols[:5]}...")
    
    # 6. Transform data
    print("\n6. Transforming data...")
    transformed_df = transform_df(original_df)
    print(f"Transformed shape: {transformed_df.shape}")
    
    # 7. Check transformed structure
    print("\n7. Transformed data structure:")
    print(f"Columns: {transformed_df.columns.tolist()[:10]}...")
    print(f"Data types: {transformed_df.dtypes[:5]}")
    
    # 8. Check timestamp conversion
    print("\n8. Timestamp converted to datetime?", 
          pd.api.types.is_datetime64_dtype(transformed_df['Timestamp']))
    
    # 9. Add derived metrics
    print("\n9. Adding derived metrics...")
    final_df = add_derived_metrics(transformed_df)
    print(f"Final shape after adding metrics: {final_df.shape}")
    print(f"New columns added: {final_df.shape[1] - transformed_df.shape[1]}")
    
    # 10. Visualize a sample of transformed data
    print("\n10. Sample of transformed data (first 3 rows):")
    pd.set_option('display.max_columns', 10)
    print(final_df.head(3))
    
    # 11. Sensor IDs found
    sensor_ids = loader.get_sensor_ids(final_df)
    print(f"\n11. Found {len(sensor_ids)} sensor IDs: {sensor_ids[:5]}...")
    
    # 12. Check for null values
    null_count = final_df.isnull().sum().sum()
    total_values = final_df.shape[0] * final_df.shape[1]
    null_percentage = (null_count / total_values) * 100
    print(f"\n12. Null values: {null_count} ({null_percentage:.2f}% of all values)")
    
    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    main() 