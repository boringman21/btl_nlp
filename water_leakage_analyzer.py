#!/usr/bin/env python3
"""
Water Leakage Analyzer - Main script for water leakage analysis

This script provides a unified interface to the water_leakage package with the following capabilities:
1. Basic analysis: Transform data and visualize correlation, flow, and pressure
2. Priority applications: Leak detection, consumption patterns, early warning, anomaly detection
3. Time-series prediction: Predict next values using LSTM models
4. Visualization: Create dashboards from previously saved results

Usage:
    python water_leakage_analyzer.py --mode <mode> [options]

Modes:
    basic-analysis: Run basic data transformation and visualization
    priority-apps: Run comprehensive analysis of priority applications
    prediction: Predict next tick values for flow and pressure
    visualize: Create visualizations from saved results
"""

import os
import argparse
import sys
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Import from water_leakage package
from water_leakage.apps.main import main as priority_apps_main
from water_leakage.apps.main import parse_args as priority_apps_parse_args
from water_leakage.apps.visualize import visualize_results, main as visualize_main
from water_leakage.data.data_loader import DataLoader
from water_leakage.data.data_transform import transform_df, add_derived_metrics, identify_potential_leaks
from water_leakage.utils.visualization import plot_flow_and_pressure, plot_correlation, plot_all_sensors, create_sensor_visualizations
from water_leakage.models.fourier import plot_fourier_approximation, analyze_frequency_components
from water_leakage.models.time_series import predict_next_tick, TimeSeriesPredictor
from water_leakage.utils.memory_utils import print_memory_usage, clear_memory


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Water Leakage Analyzer - Unified interface for water leakage analysis"
    )
    
    # General options
    parser.add_argument(
        "--mode", 
        type=str, 
        required=True,
        choices=["basic-analysis", "priority-apps", "prediction", "visualize"],
        help="Mode of operation"
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
        default="./output",
        help="Directory to save output files"
    )
    
    parser.add_argument(
        "--sensor_id", 
        type=str,
        help="Specific sensor ID to analyze (if not provided, first sensor will be used)"
    )
    
    # Basic analysis options
    parser.add_argument(
        "--num_terms", 
        type=int, 
        default=10,
        help="Number of terms to use in Fourier approximation"
    )
    
    # Prediction options
    parser.add_argument(
        "--model_dir", 
        type=str,
        default="./models",
        help="Directory to save/load trained models"
    )
    
    parser.add_argument(
        "--force_retrain",
        action="store_true",
        help="Force retraining of models even if they exist"
    )
    
    # Visualization options
    parser.add_argument(
        "--visualization", 
        action="store_true",
        help="Generate visualizations and dashboards"
    )
    
    parser.add_argument(
        "--results_file", 
        type=str,
        help="Path to the saved analysis results file (.json or .npz) for visualization mode"
    )
    
    # Priority-apps specific options
    parser.add_argument(
        "--report_dir", 
        type=str, 
        default="./report",
        help="Directory to save the analysis report (for priority-apps mode)"
    )
    
    parser.add_argument(
        "--save_results", 
        action="store_true",
        help="Save analysis results to file for later visualization"
    )
    
    return parser.parse_args()


def setup_output_directories(dirs):
    """Create output directories."""
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def run_basic_analysis(args):
    """Run basic analysis on water leak data."""
    print(f"\n=== Running Basic Analysis ===")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load data
        print("Loading data...")
        data_loader = DataLoader(args.data_path)
        data = data_loader.load_all_data()
        
        # Transform data
        print("Transforming data...")
        result_df = transform_df(data['merged_data'])
        result_df = add_derived_metrics(result_df)
        
        # Get sensor IDs
        sensor_ids = data_loader.get_sensor_ids(result_df)
        print(f"Found {len(sensor_ids)} sensors")
        
        # Create visualization directories
        visualization_dir = os.path.join(args.output_dir, "visualization")
        os.makedirs(visualization_dir, exist_ok=True)
        
        # Create visualizations for all sensors with requested directory structure
        # Use separate_metrics=True to create individual plots for each metric
        create_sensor_visualizations(result_df, sensor_ids, args.output_dir, separate_metrics=True)
        
        print(f"\nBasic analysis complete. Results saved to: {args.output_dir}")
        print(f"Visualizations saved to: {visualization_dir}")
        return 0
        
    except Exception as e:
        print(f"Error in basic analysis: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return 1


def run_priority_apps(args):
    """Run priority applications analysis."""
    print(f"\n=== Running Priority Applications Analysis ===")
    
    # Convert our args to priority_apps args
    priority_args = priority_apps_parse_args()
    priority_args.data_path = args.data_path
    priority_args.output_dir = args.output_dir
    priority_args.report_dir = args.report_dir
    priority_args.visualization = args.visualization
    priority_args.save_results = args.save_results
    priority_args.sensor_id = args.sensor_id
    
    # Run the priority apps main function
    try:
        return priority_apps_main(priority_args)
    except Exception as e:
        print(f"Error in priority applications analysis: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return 1


def run_prediction(args):
    """Run time-series prediction."""
    print(f"\n=== Running Time-Series Prediction ===")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model directory: {args.model_dir}")
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    try:
        # Load and transform data
        print("Loading data...")
        data_loader = DataLoader(args.data_path)
        data = data_loader.load_all_data()
        
        # Transform data
        print("Transforming data...")
        result_df = transform_df(data['merged_data'])
        
        # Identify sensors
        sensor_ids = data_loader.get_sensor_ids(result_df)
        
        if not sensor_ids:
            print("No sensors with complete metrics found.")
            return 1
        
        # Use specified sensor or all available sensors
        target_sensors = [args.sensor_id] if args.sensor_id else sensor_ids
        
        # Filter to valid sensors
        target_sensors = [s for s in target_sensors if s in sensor_ids]
        
        if not target_sensors:
            print(f"Specified sensor {args.sensor_id} not found or doesn't have complete metrics.")
            return 1
        
        print(f"Processing sensors: {target_sensors}")
        
        # Feature columns to predict
        feature_cols = ['Flow', 'Pressure_1', 'Pressure_2']
        
        # Initialize predictor
        predictor = TimeSeriesPredictor(window_size=24)  # 24 x 15min = 6 hours
        
        # Train models and make predictions
        all_predictions = {}
        
        for sensor_id in target_sensors:
            try:
                # Filter dataframe for the specific sensor and features
                sensor_cols = [f"{sensor_id}_{col}" for col in feature_cols]
                
                # Check if columns exist
                missing_cols = [col for col in sensor_cols if col not in result_df.columns]
                if missing_cols:
                    print(f"Missing columns for sensor {sensor_id}: {missing_cols}")
                    continue
                
                # Extract data
                sensor_df = result_df[['Timestamp'] + sensor_cols].copy()
                
                # Convert timestamp to datetime if it's not already
                sensor_df['Timestamp'] = pd.to_datetime(sensor_df['Timestamp'])
                sensor_df = sensor_df.set_index('Timestamp')
                
                # Check for sufficient data
                if len(sensor_df) < 48:  # At least 2 days of data (assuming 15-min intervals)
                    print(f"Not enough data for sensor {sensor_id}. Need at least 48 records.")
                    continue
                
                # Model path
                model_path = f"{args.model_dir}/{sensor_id}_model"
                
                # Train if needed
                if args.force_retrain or not os.path.exists(f"{model_path}_type.txt"):
                    print(f"Training model for sensor {sensor_id}...")
                    result = predictor.fit(sensor_df)
                    
                    if "error" in result:
                        print(f"Error training model for sensor {sensor_id}: {result['error']}")
                        continue
                    
                    # Save model
                    predictor.save_model(model_path)
                    print(f"Model trained and saved for sensor {sensor_id}")
                else:
                    print(f"Loading existing model for sensor {sensor_id}...")
                    predictor.load_model(model_path)
                
                # Make prediction
                print(f"Predicting next tick for sensor {sensor_id}...")
                prediction = predict_next_tick(
                    result_df,
                    sensor_id,
                    feature_cols,
                    model_path=args.model_dir
                )
                
                # Get last timestamp and calculate next timestamp
                last_timestamp = result_df['Timestamp'].max()
                from datetime import timedelta
                next_timestamp = last_timestamp + timedelta(minutes=15)
                
                # Store prediction
                all_predictions[sensor_id] = {
                    "timestamp": next_timestamp.isoformat(),
                    "features": prediction
                }
                
                # Print prediction
                print(f"Sensor {sensor_id} - Next tick prediction:")
                for feature, value in prediction.items():
                    print(f"  {feature}: {value:.4f}")
                
                # Generate visualization if requested
                if args.visualization:
                    # Create prediction visualization
                    sensor_df = result_df[['Timestamp'] + sensor_cols].copy()
                    plot_predictions(
                        sensor_df, 
                        prediction, 
                        sensor_id,
                        next_timestamp,
                        os.path.join(args.output_dir, f"{sensor_id}_prediction.png")
                    )
                    print(f"Prediction plot saved to: {args.output_dir}/{sensor_id}_prediction.png")
            
            except Exception as e:
                print(f"Error making prediction for sensor {sensor_id}: {str(e)}")
                import traceback
                print(traceback.format_exc())
        
        # Save predictions to file
        import json
        prediction_file = os.path.join(args.output_dir, f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(prediction_file, 'w') as f:
            json.dump(all_predictions, f, indent=2)
        
        print(f"Predictions saved to: {prediction_file}")
        
        return 0
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return 1


def run_visualization(args):
    """Run visualization of saved results."""
    print(f"\n=== Running Visualization ===")
    
    if not args.results_file:
        print("Error: --results_file is required for visualization mode")
        return 1
    
    print(f"Results file: {args.results_file}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Run the visualize main function
        return visualize_main(args.results_file, args.output_dir, args.sensor_id)
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return 1


def main():
    """Main function to run the water leakage analyzer."""
    # Parse command line arguments
    args = parse_args()
    
    # Print header
    print("\n===== Water Leakage Analyzer =====")
    print(f"Mode: {args.mode}")
    
    # Run selected mode
    if args.mode == "basic-analysis":
        return run_basic_analysis(args)
    elif args.mode == "priority-apps":
        return run_priority_apps(args)
    elif args.mode == "prediction":
        return run_prediction(args)
    elif args.mode == "visualize":
        return run_visualization(args)
    else:
        print(f"Unknown mode: {args.mode}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 