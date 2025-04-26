#!/usr/bin/env python
"""
Sample script for analyzing water leak data using the water_leakage package.

Before running this script, ensure:
1. You have installed and activated a virtual environment
2. The water_leakage package is installed
3. Data files are available at the specified path

Usage:
    python run_sample.py [--data_path PATH] [--output_dir PATH] [--sensor_id ID] [--num_terms N]
"""

import os
import argparse
import logging
import traceback
import matplotlib.pyplot as plt
from water_leakage.data.data_loader import DataLoader
from water_leakage.data.data_transform import transform_df, add_derived_metrics, identify_potential_leaks
from water_leakage.utils.visualization import plot_flow_and_pressure, plot_correlation, plot_all_sensors
from water_leakage.models.fourier import plot_fourier_approximation, analyze_frequency_components
from water_leakage.utils.memory_utils import print_memory_usage, clear_memory


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze water leak data')
    parser.add_argument('--data_path', type=str, default="./LeakDataset/Logger_Data_2024_Bau_Bang-2",
                        help='Path to the data directory')
    parser.add_argument('--output_dir', type=str, default="./output_plots",
                        help='Directory to save output plots')
    parser.add_argument('--sensor_id', type=str, default=None,
                        help='Specific sensor ID to analyze (if not provided, first sensor will be used)')
    parser.add_argument('--num_terms', type=int, default=10,
                        help='Number of terms to use in Fourier approximation')
    return parser.parse_args()


def load_and_transform_data(data_path):
    """
    Load and transform the data.
    
    Args:
        data_path (str): Path to the data directory
        
    Returns:
        tuple: (transformed_df, sensor_ids, data_loader)
    """
    logging.info("Loading data from: %s", data_path)
    data_loader = DataLoader(data_path)
    data = data_loader.load_all_data()
    logging.info("Loaded %d datasets", len(data))
    
    logging.info("Transforming data...")
    result_df = transform_df(data['merged_data'])
    logging.info("Data transformed to shape: %s", result_df.shape)
    
    logging.info("Adding derived metrics...")
    result_df = add_derived_metrics(result_df)
    
    # Get list of sensor IDs
    sensor_ids = data_loader.get_sensor_ids(result_df)
    logging.info("Found %d sensors", len(sensor_ids))
    
    return result_df, sensor_ids, data_loader


def analyze_sensor(result_df, sensor_id, output_dir, num_terms=10):
    """
    Analyze a specific sensor.
    
    Args:
        result_df (pd.DataFrame): Transformed DataFrame
        sensor_id (str): Sensor ID to analyze
        output_dir (str): Directory to save output plots
        num_terms (int): Number of terms for Fourier approximation
        
    Returns:
        bool: True if successful, False otherwise
    """
    logging.info("Analyzing sensor: %s", sensor_id)
    
    # Columns for this sensor
    flow_col = f"{sensor_id}_Flow"
    pressure_1_col = f"{sensor_id}_Pressure_1"
    pressure_2_col = f"{sensor_id}_Pressure_2"
    pressure_diff_col = f"{sensor_id}_Pressure_Diff"
    
    # Check if required columns exist
    if flow_col not in result_df.columns:
        logging.warning("Flow data missing for sensor %s", sensor_id)
        return False
    
    # Plot flow and pressure
    logging.info("Plotting flow and pressure data...")
    fig = plot_flow_and_pressure(result_df, sensor_id, save_dir=output_dir)
    if fig:
        plt.close(fig)
    
    # Plot correlation matrix
    logging.info("Plotting correlation matrix...")
    fig = plot_correlation(result_df, sensor_id, save_dir=output_dir)
    if fig:
        plt.close(fig)
    
    # Fourier analysis
    if flow_col in result_df.columns:
        logging.info("Performing Fourier analysis with %d terms...", num_terms)
        fig = plot_fourier_approximation(
            result_df['Timestamp'],
            result_df[flow_col].values,
            'Flow',
            sensor_id,
            save_dir=output_dir,
            num_terms=num_terms
        )
        if fig:
            plt.close(fig)
    
    # Check for potential leaks
    if pressure_diff_col in result_df.columns:
        logging.info("Checking for potential leaks...")
        leak_events = identify_potential_leaks(result_df, threshold_diff=0.5)
        if sensor_id in leak_events:
            logging.info("Found %d potential leak events for sensor %s", 
                        len(leak_events[sensor_id]), sensor_id)
    
    return True


def analyze_all_sensors(result_df, sensor_ids, output_dir):
    """
    Create plots comparing all sensors.
    
    Args:
        result_df (pd.DataFrame): Transformed DataFrame
        sensor_ids (list): List of sensor IDs
        output_dir (str): Directory to save output plots
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not sensor_ids:
        logging.warning("No sensors to analyze")
        return False
    
    logging.info("Creating plots comparing all sensors...")
    
    # Plot flow for all sensors
    fig = plot_all_sensors(result_df, metric='Flow', save_dir=output_dir)
    if fig:
        plt.close(fig)
        
    # Plot pressure for all sensors
    fig = plot_all_sensors(result_df, metric='Pressure_2', save_dir=output_dir)
    if fig:
        plt.close(fig)
    
    # Plot pressure difference for all sensors
    fig = plot_all_sensors(result_df, metric='Pressure_Diff', save_dir=output_dir)
    if fig:
        plt.close(fig)
    
    return True


def main():
    """Main execution function."""
    # Set up logging
    setup_logging()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Track memory usage
    print_memory_usage()
    
    try:
        # Load and transform data
        result_df, sensor_ids, data_loader = load_and_transform_data(args.data_path)
        
        if not sensor_ids:
            logging.error("No sensors found in the data")
            return
        
        # Determine which sensor to analyze
        sensor_id_to_analyze = args.sensor_id if args.sensor_id else sensor_ids[0]
        if sensor_id_to_analyze not in sensor_ids:
            logging.warning("Specified sensor %s not found. Using %s instead.", 
                          sensor_id_to_analyze, sensor_ids[0])
            sensor_id_to_analyze = sensor_ids[0]
        
        # Analyze the selected sensor
        analyze_sensor(result_df, sensor_id_to_analyze, args.output_dir, args.num_terms)
        
        # Create plots comparing all sensors
        analyze_all_sensors(result_df, sensor_ids, args.output_dir)
        
        logging.info("Analysis complete. Plots saved to: %s", args.output_dir)
    
    except FileNotFoundError as e:
        logging.error("File not found: %s", str(e))
    except ValueError as e:
        logging.error("Value error: %s", str(e))
    except Exception as e:
        logging.error("Unexpected error: %s", str(e))
        logging.debug(traceback.format_exc())
    
    finally:
        # Clean up and release memory
        logging.info("Cleaning up and releasing memory...")
        clear_memory()
        print_memory_usage()


if __name__ == "__main__":
    main() 