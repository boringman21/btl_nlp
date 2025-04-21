#!/usr/bin/env python3
"""
Script to predict the next 15-minute tick based on historical data.
Uses LSTM models to predict flow and pressure values for selected sensors.
"""

import os
import argparse
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path

from water_leakage.data.data_loader import load_data
from water_leakage.data.data_transform import transform_df
from water_leakage.models.time_series import predict_next_tick
from water_leakage.utils.visualization import plot_predictions


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict the next 15-minute tick based on historical data"
    )
    
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="./LeakDataset/Logger_Data_2024_Bau_Bang-2",
        help="Path to the logger data directory"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./prediction_results",
        help="Directory to save prediction results"
    )
    
    parser.add_argument(
        "--sensor_id", 
        type=str,
        help="Specific sensor ID to predict for. If not provided, predictions will be made for all sensors with complete data."
    )
    
    parser.add_argument(
        "--model_dir", 
        type=str,
        default="./models",
        help="Directory to save/load trained models"
    )
    
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Generate visualization of the prediction"
    )
    
    return parser.parse_args()


def main():
    """Main function to run the prediction."""
    # Parse command line arguments
    args = parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    print(f"Loading data from: {args.data_path}")
    
    # Load and transform data
    merged_df = load_data(args.data_path)
    if merged_df is None or merged_df.empty:
        print("Error: Could not load data.")
        return 1
    
    result_df = transform_df(merged_df)
    
    # Identify sensors that have all three metrics (Flow, Pressure_1, Pressure_2)
    def get_sensor_ids(df):
        filtered_columns = df.columns[df.columns.str.endswith('_Pressure_2')]
        return [col.split('_')[0] for col in filtered_columns]
    
    sensor_ids = get_sensor_ids(result_df)
    
    if not sensor_ids:
        print("Error: No sensors with complete metrics found.")
        return 1
    
    # Use specified sensor or all available sensors
    target_sensors = [args.sensor_id] if args.sensor_id else sensor_ids
    
    # Filter to valid sensors
    target_sensors = [s for s in target_sensors if s in sensor_ids]
    
    if not target_sensors:
        print(f"Error: Specified sensor {args.sensor_id} not found or doesn't have complete metrics.")
        return 1
    
    print(f"Making predictions for sensors: {target_sensors}")
    
    # Make predictions
    all_predictions = {}
    feature_cols = ['Flow', 'Pressure_1', 'Pressure_2']
    
    for sensor_id in target_sensors:
        try:
            prediction = predict_next_tick(
                result_df,
                sensor_id,
                feature_cols,
                model_path=args.model_dir
            )
            
            # Store prediction along with timestamp
            last_timestamp = result_df['Timestamp'].max()
            next_timestamp = last_timestamp + timedelta(minutes=15)
            
            all_predictions[sensor_id] = {
                "timestamp": next_timestamp.isoformat(),
                "features": prediction
            }
            
            print(f"Sensor {sensor_id} - Next tick prediction:")
            for feature, value in prediction.items():
                print(f"  {feature}: {value:.4f}")
            
            # Generate visualization if requested
            if args.visualize:
                # Get historical data for this sensor
                sensor_cols = [f"{sensor_id}_{col}" for col in feature_cols]
                if all(col in result_df.columns for col in sensor_cols):
                    # Create prediction visualization
                    sensor_df = result_df[['Timestamp'] + sensor_cols].copy()
                    plot_predictions(
                        sensor_df, 
                        prediction, 
                        sensor_id,
                        next_timestamp,
                        os.path.join(args.output_dir, f"{sensor_id}_prediction.png")
                    )
        
        except Exception as e:
            print(f"Error making prediction for sensor {sensor_id}: {str(e)}")
    
    # Save predictions to file
    prediction_file = os.path.join(args.output_dir, f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(prediction_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    
    print(f"Predictions saved to: {prediction_file}")
    
    return 0


if __name__ == "__main__":
    exit(main()) 