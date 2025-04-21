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
import logging

from water_leakage.data.data_loader import load_data
from water_leakage.data.data_transform import transform_df
from water_leakage.models.time_series import predict_next_tick, TimeSeriesPredictor
from water_leakage.utils.visualization import plot_predictions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    
    parser.add_argument(
        "--force_retrain",
        action="store_true",
        help="Force retraining of models even if they exist"
    )
    
    return parser.parse_args()


def train_model_for_sensor(result_df, sensor_id, feature_cols, model_dir, force_retrain=False):
    """
    Train a model for a specific sensor
    
    Args:
        result_df (pd.DataFrame): The processed data
        sensor_id (str): The sensor ID to train for
        feature_cols (list): The feature columns to use
        model_dir (str): Directory to save the model
        force_retrain (bool): Whether to force retraining even if a model exists
        
    Returns:
        bool: True if training was successful, False otherwise
    """
    # Create necessary directories
    os.makedirs(model_dir, exist_ok=True)
    
    # Filter dataframe for the specific sensor and features
    sensor_cols = [f"{sensor_id}_{col}" for col in feature_cols]
    
    # Check if columns exist
    missing_cols = [col for col in sensor_cols if col not in result_df.columns]
    if missing_cols:
        logger.error(f"Missing columns for sensor {sensor_id}: {missing_cols}")
        return False
    
    # Extract data
    sensor_df = result_df[['Timestamp'] + sensor_cols].copy()
    
    # Convert timestamp to datetime if it's not already
    sensor_df['Timestamp'] = pd.to_datetime(sensor_df['Timestamp'])
    sensor_df = sensor_df.set_index('Timestamp')
    
    # Check for sufficient data
    if len(sensor_df) < 48:  # At least 2 days of data (assuming 15-min intervals)
        logger.error(f"Not enough data for sensor {sensor_id}. Need at least 48 records.")
        return False
    
    # Create predictor
    predictor = TimeSeriesPredictor(window_size=24)  # 24 x 15min = 6 hours
    
    # Check if model exists and whether to retrain
    model_path = f"{model_dir}/{sensor_id}_model"
    scaler_path = f"{model_dir}/{sensor_id}_scaler.pkl"
    
    # Train a new model if forced, or if no model exists
    if force_retrain or not (os.path.exists(f"{model_path}_type.txt") or 
                             (os.path.exists(model_path) and os.path.exists(scaler_path))):
        logger.info(f"Training new model for sensor {sensor_id}")
        try:
            # Train the model
            result = predictor.fit(sensor_df)
            
            # Check if training was successful
            if "error" in result:
                logger.error(f"Error training model for sensor {sensor_id}: {result['error']}")
                return False
            
            # Save the model
            predictor.save_model(model_path)
            logger.info(f"Model trained and saved for sensor {sensor_id}")
            return True
        except Exception as e:
            logger.error(f"Exception during model training for sensor {sensor_id}: {str(e)}")
            return False
    else:
        logger.info(f"Model already exists for sensor {sensor_id}")
        return True


def main():
    """Main function to run the prediction."""
    # Parse command line arguments
    args = parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    logger.info(f"Loading data from: {args.data_path}")
    
    # Load and transform data
    merged_df = load_data(args.data_path)
    if merged_df is None or merged_df.empty:
        logger.error("Could not load data.")
        return 1
    
    result_df = transform_df(merged_df)
    
    # Identify sensors that have all three metrics (Flow, Pressure_1, Pressure_2)
    def get_sensor_ids(df):
        filtered_columns = df.columns[df.columns.str.endswith('_Pressure_2')]
        return [col.split('_')[0] for col in filtered_columns]
    
    sensor_ids = get_sensor_ids(result_df)
    
    if not sensor_ids:
        logger.error("No sensors with complete metrics found.")
        return 1
    
    # Use specified sensor or all available sensors
    target_sensors = [args.sensor_id] if args.sensor_id else sensor_ids
    
    # Filter to valid sensors
    target_sensors = [s for s in target_sensors if s in sensor_ids]
    
    if not target_sensors:
        logger.error(f"Specified sensor {args.sensor_id} not found or doesn't have complete metrics.")
        return 1
    
    logger.info(f"Processing sensors: {target_sensors}")
    
    # Feature columns to predict
    feature_cols = ['Flow', 'Pressure_1', 'Pressure_2']
    
    # First, train models for all sensors if needed
    for sensor_id in target_sensors:
        train_model_for_sensor(result_df, sensor_id, feature_cols, args.model_dir, args.force_retrain)
    
    # Make predictions
    all_predictions = {}
    
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
            
            logger.info(f"Sensor {sensor_id} - Next tick prediction:")
            for feature, value in prediction.items():
                logger.info(f"  {feature}: {value:.4f}")
            
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
            logger.error(f"Error making prediction for sensor {sensor_id}: {str(e)}")
    
    # Save predictions to file
    prediction_file = os.path.join(args.output_dir, f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(prediction_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    
    logger.info(f"Predictions saved to: {prediction_file}")
    
    return 0


if __name__ == "__main__":
    exit(main()) 