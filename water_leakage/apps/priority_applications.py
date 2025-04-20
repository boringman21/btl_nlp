#!/usr/bin/env python3
"""
Module for analyzing water leak data focused on priority applications:
1. Leak Detection
2. Water Consumption Pattern Prediction
3. Early Warning System
4. Anomaly Detection and Prediction
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from water_leakage.data.data_loader import DataLoader
from water_leakage.data.data_transform import transform_df, add_derived_metrics, identify_potential_leaks
from water_leakage.utils.visualization import plot_flow_and_pressure, plot_correlation, plot_all_sensors
from water_leakage.models.fourier import plot_fourier_approximation, analyze_frequency_components
from water_leakage.utils.memory_utils import print_memory_usage, clear_memory


def setup_directories(data_path, output_dir, report_dir):
    """Create output directories if they don't exist."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    
    # Create subdirectories for each analysis type
    for subdir in ['leak_detection', 'consumption_patterns', 'early_warning', 'anomaly_detection']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)


def load_and_prepare_data(data_path):
    """Load and prepare data for analysis."""
    print("Loading and preparing data...")
    
    # Load data
    loader = DataLoader(data_path)
    data = loader.load_all_data()
    
    # Transform data
    transformed_df = transform_df(data['merged_data'])
    final_df = add_derived_metrics(transformed_df)
    
    # Get sensor IDs
    sensor_ids = loader.get_sensor_ids(final_df)
    
    return final_df, sensor_ids, loader


def analyze_leak_detection(df, sensor_ids, output_dir):
    """
    Analyze data for leak detection.
    
    Args:
        df (pd.DataFrame): Processed DataFrame
        sensor_ids (List[str]): List of sensor IDs
        output_dir (str): Directory to save results
        
    Returns:
        Dict: Results including leak events and statistics
    """
    print("\n=== Leak Detection Analysis ===")
    save_dir = os.path.join(output_dir, 'leak_detection')
    
    # Dictionary to store results
    results = {
        'leak_events': {},
        'sensor_summaries': {},
        'total_leaks': 0
    }
    
    # Identify potential leaks (try different thresholds)
    thresholds = [0.5, 1.0, 1.5]
    for threshold in thresholds:
        print(f"Analyzing with threshold: {threshold}")
        leak_events = identify_potential_leaks(df, threshold_diff=threshold)
        
        if threshold == 1.0:  # Store the medium threshold results
            results['leak_events'] = leak_events
            results['total_leaks'] = sum(len(events) for events in leak_events.values())
    
    # Analyze each sensor
    for sensor_id in sensor_ids:
        print(f"Analyzing sensor: {sensor_id}")
        # Create visualization
        fig = plot_flow_and_pressure(df, sensor_id, save_dir=save_dir)
        if fig:
            plt.close(fig)
        
        # Calculate statistics for this sensor
        flow_col = f'{sensor_id}_Flow'
        pressure_diff_col = f'{sensor_id}_Pressure_Diff'
        
        # Summary statistics
        sensor_summary = {}
        if flow_col in df.columns:
            sensor_summary['flow_mean'] = df[flow_col].mean()
            sensor_summary['flow_std'] = df[flow_col].std()
        
        if pressure_diff_col in df.columns:
            sensor_summary['pressure_diff_mean'] = df[pressure_diff_col].mean()
            sensor_summary['pressure_diff_std'] = df[pressure_diff_col].std()
            
            # Count of leak events for this sensor
            if sensor_id in results['leak_events']:
                sensor_summary['leak_count'] = len(results['leak_events'][sensor_id])
            else:
                sensor_summary['leak_count'] = 0
        
        results['sensor_summaries'][sensor_id] = sensor_summary
    
    # Print summary
    print(f"Total potential leak events detected: {results['total_leaks']}")
    print("Leak events by sensor:")
    for sensor_id, summary in results['sensor_summaries'].items():
        if 'leak_count' in summary:
            print(f"  - Sensor {sensor_id}: {summary['leak_count']} events")
    
    return results


def analyze_consumption_patterns(df, sensor_ids, output_dir):
    """
    Analyze water consumption patterns.
    
    Args:
        df (pd.DataFrame): Processed DataFrame
        sensor_ids (List[str]): List of sensor IDs
        output_dir (str): Directory to save results
        
    Returns:
        Dict: Results including patterns and statistics
    """
    print("\n=== Water Consumption Pattern Analysis ===")
    save_dir = os.path.join(output_dir, 'consumption_patterns')
    
    # Add timestamp components for pattern analysis
    df = df.copy()
    df['hour'] = df['Timestamp'].dt.hour
    df['day'] = df['Timestamp'].dt.day_name()
    
    # Dictionary to store results
    results = {
        'hourly_patterns': {},
        'daily_patterns': {},
        'sensor_stats': {}
    }
    
    # Analyze each sensor
    for sensor_id in sensor_ids:
        flow_col = f'{sensor_id}_Flow'
        if flow_col not in df.columns:
            continue
            
        print(f"Analyzing consumption patterns for sensor: {sensor_id}")
        
        # Create hourly and daily aggregates
        hourly_avg = df.groupby('hour')[flow_col].mean()
        daily_avg = df.groupby('day')[flow_col].mean()
        
        # Store results
        results['hourly_patterns'][sensor_id] = hourly_avg.to_dict()
        results['daily_patterns'][sensor_id] = daily_avg.to_dict()
        
        # Calculate statistics
        results['sensor_stats'][sensor_id] = {
            'peak_hour': hourly_avg.idxmax(),
            'min_hour': hourly_avg.idxmin(),
            'peak_value': hourly_avg.max(),
            'min_value': hourly_avg.min(),
            'variance': hourly_avg.var()
        }
        
        # Create visualizations
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot hourly patterns
        hourly_avg.plot(kind='bar', ax=ax1)
        ax1.set_title(f'Average Flow by Hour (Sensor {sensor_id})')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Average Flow')
        
        # Plot daily patterns
        daily_avg.plot(kind='bar', ax=ax2)
        ax2.set_title(f'Average Flow by Day (Sensor {sensor_id})')
        ax2.set_xlabel('Day of Week')
        ax2.set_ylabel('Average Flow')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'sensor_{sensor_id}_patterns.png'))
        plt.close()
    
    # Print summary
    print("Consumption Pattern Summary:")
    for sensor_id, stats in results['sensor_stats'].items():
        print(f"  - Sensor {sensor_id}:")
        print(f"    Peak hour: {stats['peak_hour']} ({stats['peak_value']:.2f})")
        print(f"    Minimum hour: {stats['min_hour']} ({stats['min_value']:.2f})")
    
    return results


def analyze_early_warning(df, sensor_ids, leak_results, output_dir):
    """
    Analyze data for early warning system.
    
    Args:
        df (pd.DataFrame): Processed DataFrame
        sensor_ids (List[str]): List of sensor IDs
        leak_results (Dict): Results from leak detection analysis
        output_dir (str): Directory to save results
        
    Returns:
        Dict: Results including early warning indicators
    """
    print("\n=== Early Warning System Analysis ===")
    save_dir = os.path.join(output_dir, 'early_warning')
    
    # Dictionary to store results
    results = {
        'warning_indicators': {},
        'sensitivity_analysis': {}
    }
    
    # Calculate rolling statistics to detect early warning signs
    window_sizes = [5, 10, 20]
    
    for sensor_id in sensor_ids:
        pressure_diff_col = f'{sensor_id}_Pressure_Diff'
        flow_col = f'{sensor_id}_Flow'
        
        if pressure_diff_col not in df.columns or flow_col not in df.columns:
            continue
            
        print(f"Analyzing early warning indicators for sensor: {sensor_id}")
        
        # Create dataframe for this sensor's analysis
        sensor_df = df[['Timestamp', pressure_diff_col, flow_col]].copy()
        
        # Warning indicators for different window sizes
        indicators = {}
        
        for window in window_sizes:
            # Calculate rolling statistics
            sensor_df[f'roll_mean_{window}'] = sensor_df[pressure_diff_col].rolling(window=window).mean()
            sensor_df[f'roll_std_{window}'] = sensor_df[pressure_diff_col].rolling(window=window).std()
            
            # Calculate deviation from rolling mean
            sensor_df[f'deviation_{window}'] = (
                sensor_df[pressure_diff_col] - sensor_df[f'roll_mean_{window}']
            ) / sensor_df[f'roll_std_{window}']
            
            # Consider values exceeding 2 standard deviations as warning signs
            warning_mask = abs(sensor_df[f'deviation_{window}']) > 2
            
            # Count number of warnings
            warning_count = warning_mask.sum()
            
            # Store results
            indicators[window] = {
                'warning_count': warning_count,
                'warning_percentage': (warning_count / len(sensor_df)) * 100
            }
        
        results['warning_indicators'][sensor_id] = indicators
        
        # Find the most sensitive and specific window size
        sensitivity_dict = {}
        for window in window_sizes:
            # If we have leak data for this sensor
            if sensor_id in leak_results['leak_events']:
                # Get leak events
                leak_events = leak_results['leak_events'][sensor_id]
                leak_timestamps = leak_events['Timestamp'].tolist()
                
                # Calculate how many leaks were preceded by warnings (within 5 time periods)
                warnings_before_leaks = 0
                
                for leak_time in leak_timestamps:
                    # Get data before this leak
                    before_leak = sensor_df[sensor_df['Timestamp'] < leak_time].tail(5)
                    
                    # Check if any warning was triggered
                    if any(abs(before_leak[f'deviation_{window}']) > 2):
                        warnings_before_leaks += 1
                
                if len(leak_timestamps) > 0:
                    detection_rate = warnings_before_leaks / len(leak_timestamps) * 100
                else:
                    detection_rate = 0
                    
                sensitivity_dict[window] = {
                    'detection_rate': detection_rate,
                    'false_positive_rate': indicators[window]['warning_percentage']
                }
        
        results['sensitivity_analysis'][sensor_id] = sensitivity_dict
        
        # Create visualization of warning indicators
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot pressure difference
        ax.plot(sensor_df['Timestamp'], sensor_df[pressure_diff_col], label='Pressure Difference')
        
        # Plot rolling mean with the middle window size
        middle_window = window_sizes[len(window_sizes)//2]
        ax.plot(
            sensor_df['Timestamp'], 
            sensor_df[f'roll_mean_{middle_window}'], 
            label=f'Rolling Mean (window={middle_window})',
            color='green'
        )
        
        # Mark warning points
        warning_points = sensor_df[abs(sensor_df[f'deviation_{middle_window}']) > 2]
        ax.scatter(
            warning_points['Timestamp'], 
            warning_points[pressure_diff_col],
            color='red',
            label='Warning Points',
            zorder=5
        )
        
        ax.set_title(f'Early Warning Indicators (Sensor {sensor_id})')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Pressure Difference')
        ax.legend()
        
        plt.savefig(os.path.join(save_dir, f'sensor_{sensor_id}_early_warning.png'))
        plt.close()
    
    # Print summary
    print("Early Warning System Summary:")
    for sensor_id, indicators in results['warning_indicators'].items():
        print(f"  - Sensor {sensor_id}:")
        for window, stats in indicators.items():
            print(f"    Window {window}: {stats['warning_count']} warnings ({stats['warning_percentage']:.2f}%)")
    
    return results


def analyze_anomalies(df, sensor_ids, output_dir):
    """
    Analyze and predict anomalies in the data.
    
    Args:
        df (pd.DataFrame): Processed DataFrame
        sensor_ids (List[str]): List of sensor IDs
        output_dir (str): Directory to save results
        
    Returns:
        Dict: Results including anomaly detection
    """
    print("\n=== Anomaly Detection and Prediction ===")
    save_dir = os.path.join(output_dir, 'anomaly_detection')
    
    # Dictionary to store results
    results = {
        'anomalies': {},
        'fourier_analysis': {}
    }
    
    # For each sensor
    for sensor_id in sensor_ids:
        flow_col = f'{sensor_id}_Flow'
        
        if flow_col not in df.columns:
            continue
            
        print(f"Analyzing anomalies for sensor: {sensor_id}")
        
        # Statistical anomaly detection
        # Z-score method: values more than 3 standard deviations from mean
        mean = df[flow_col].mean()
        std = df[flow_col].std()
        
        # Identify anomalies
        anomaly_mask = abs(df[flow_col] - mean) > 3 * std
        anomalies = df[anomaly_mask].copy()
        
        # Store results
        results['anomalies'][sensor_id] = {
            'count': len(anomalies),
            'percentage': (len(anomalies) / len(df)) * 100,
            'min_value': anomalies[flow_col].min() if not anomalies.empty else None,
            'max_value': anomalies[flow_col].max() if not anomalies.empty else None
        }
        
        # Fourier analysis for pattern detection
        try:
            # Create a copy with only valid flow values
            valid_mask = df[flow_col].notna()
            temp_df = df[valid_mask].copy()
            
            # Perform Fourier analysis
            fig = plot_fourier_approximation(
                temp_df['Timestamp'],
                temp_df[flow_col].values,
                'Flow',
                sensor_id,
                save_dir=save_dir,
                num_terms=5
            )
            
            if fig:
                plt.close(fig)
                
            # Store results of frequency components
            frequency_results = analyze_frequency_components(
                temp_df['Timestamp'],
                temp_df[flow_col].values,
                num_terms=5
            )
            
            if frequency_results:
                results['fourier_analysis'][sensor_id] = frequency_results
        
        except Exception as e:
            print(f"Error in Fourier analysis for sensor {sensor_id}: {str(e)}")
        
        # Create visualization of anomalies
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot flow data
        ax.plot(df['Timestamp'], df[flow_col], label='Flow', alpha=0.7)
        
        # Mark anomalies
        ax.scatter(
            df[anomaly_mask]['Timestamp'],
            df[anomaly_mask][flow_col],
            color='red',
            label='Anomalies',
            zorder=5
        )
        
        # Add threshold lines
        ax.axhline(mean + 3*std, color='orange', linestyle='--', label='Upper Threshold')
        ax.axhline(mean - 3*std, color='orange', linestyle='--', label='Lower Threshold')
        
        ax.set_title(f'Flow Anomalies (Sensor {sensor_id})')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Flow')
        ax.legend()
        
        plt.savefig(os.path.join(save_dir, f'sensor_{sensor_id}_anomalies.png'))
        plt.close()
    
    # Print summary
    print("Anomaly Detection Summary:")
    for sensor_id, stats in results['anomalies'].items():
        print(f"  - Sensor {sensor_id}: {stats['count']} anomalies ({stats['percentage']:.2f}%)")
    
    return results


def generate_report(leak_results, consumption_results, warning_results, anomaly_results, sensor_ids, report_dir, output_dir):
    """
    Generate a comprehensive report combining all analyses.
    
    Args:
        leak_results (Dict): Results from leak detection analysis
        consumption_results (Dict): Results from consumption pattern analysis
        warning_results (Dict): Results from early warning system analysis
        anomaly_results (Dict): Results from anomaly detection
        sensor_ids (List[str]): List of sensor IDs
        report_dir (str): Directory to save the report
        output_dir (str): Directory with analysis results
    """
    print("\n=== Generating Report ===")
    
    # Create report file
    report_file = os.path.join(report_dir, f"water_leakage_analysis_report_{datetime.now().strftime('%Y%m%d')}.md")
    
    with open(report_file, 'w') as f:
        # Write report header
        f.write("# Water Leakage Analysis Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n\n")
        f.write("## Executive Summary\n\n")
        
        # Executive summary
        total_leaks = leak_results.get('total_leaks', 0)
        f.write(f"This analysis processed data from {len(sensor_ids)} sensors and detected {total_leaks} potential leak events.\n\n")
        
        # Key findings
        f.write("### Key Findings\n\n")
        
        # Leak detection findings
        f.write("#### Leak Detection\n")
        f.write(f"- Total potential leak events: {total_leaks}\n")
        for sensor_id, summary in leak_results.get('sensor_summaries', {}).items():
            if 'leak_count' in summary and summary['leak_count'] > 0:
                f.write(f"- Sensor {sensor_id}: {summary['leak_count']} potential leak events\n")
        f.write("\n")
        
        # Consumption pattern findings
        f.write("#### Water Consumption Patterns\n")
        for sensor_id, stats in consumption_results.get('sensor_stats', {}).items():
            f.write(f"- Sensor {sensor_id}:\n")
            f.write(f"  - Peak usage at hour {stats['peak_hour']}\n")
            f.write(f"  - Minimum usage at hour {stats['min_hour']}\n")
            f.write(f"  - Flow variation: {stats['variance']:.2f}\n")
        f.write("\n")
        
        # Early warning findings
        f.write("#### Early Warning System\n")
        for sensor_id, indicators in warning_results.get('warning_indicators', {}).items():
            # Choose middle window size for reporting
            window_sizes = list(indicators.keys())
            if window_sizes:
                middle_window = window_sizes[len(window_sizes)//2]
                stats = indicators[middle_window]
                f.write(f"- Sensor {sensor_id}:\n")
                f.write(f"  - {stats['warning_count']} early warning indicators ({stats['warning_percentage']:.2f}%)\n")
                
                # Add sensitivity if available
                if sensor_id in warning_results.get('sensitivity_analysis', {}) and middle_window in warning_results['sensitivity_analysis'][sensor_id]:
                    sensitivity = warning_results['sensitivity_analysis'][sensor_id][middle_window]
                    f.write(f"  - Detection rate: {sensitivity['detection_rate']:.2f}%\n")
        f.write("\n")
        
        # Anomaly detection findings
        f.write("#### Anomaly Detection\n")
        for sensor_id, stats in anomaly_results.get('anomalies', {}).items():
            f.write(f"- Sensor {sensor_id}:\n")
            f.write(f"  - {stats['count']} anomalies detected ({stats['percentage']:.2f}%)\n")
            if stats['min_value'] is not None and stats['max_value'] is not None:
                f.write(f"  - Anomaly range: {stats['min_value']:.2f} to {stats['max_value']:.2f}\n")
        f.write("\n")
        
        # Detailed results sections
        f.write("## Detailed Analysis Results\n\n")
        
        # Add sections for each analysis type
        f.write("### 1. Leak Detection Analysis\n\n")
        f.write("This analysis identifies potential leak events by detecting sudden changes in pressure differences.\n\n")
        f.write(f"![Sample leak detection image]({os.path.relpath(output_dir, report_dir)}/leak_detection/sensor_ID_plot.png)\n\n")
        
        f.write("### 2. Water Consumption Pattern Analysis\n\n")
        f.write("This analysis examines patterns in water usage over time to identify peak usage periods and trends.\n\n")
        f.write(f"![Sample consumption pattern image]({os.path.relpath(output_dir, report_dir)}/consumption_patterns/sensor_ID_patterns.png)\n\n")
        
        f.write("### 3. Early Warning System Analysis\n\n")
        f.write("This analysis detects early warning signs that may indicate potential leaks before they become severe.\n\n")
        f.write(f"![Sample early warning image]({os.path.relpath(output_dir, report_dir)}/early_warning/sensor_ID_early_warning.png)\n\n")
        
        f.write("### 4. Anomaly Detection and Prediction\n\n")
        f.write("This analysis identifies unusual patterns in the data that deviate from normal behavior.\n\n")
        f.write(f"![Sample anomaly detection image]({os.path.relpath(output_dir, report_dir)}/anomaly_detection/sensor_ID_anomalies.png)\n\n")
        f.write(f"![Sample Fourier analysis image]({os.path.relpath(output_dir, report_dir)}/anomaly_detection/ID/ID_Flow_fourier.png)\n\n")
        
        # Recommendations section
        f.write("## Recommendations\n\n")
        
        f.write("### Leak Detection\n")
        f.write("- Investigate sensors with the highest number of detected leak events\n")
        f.write("- Deploy maintenance teams to areas with consistent pressure anomalies\n")
        f.write("- Optimize pressure thresholds for more accurate leak detection\n\n")
        
        f.write("### Water Consumption\n")
        f.write("- Adjust water supply during peak usage times identified in the analysis\n")
        f.write("- Develop targeted conservation measures for high-consumption periods\n")
        f.write("- Consider implementing time-of-day pricing based on usage patterns\n\n")
        
        f.write("### Early Warning System\n")
        f.write("- Implement automated alerts based on the early warning thresholds\n")
        f.write("- Focus monitoring resources on sensors with high warning rates\n")
        f.write("- Consider decreasing response time thresholds for critical areas\n\n")
        
        f.write("### Anomaly Detection\n")
        f.write("- Investigate recurring anomalies for potential system design issues\n")
        f.write("- Develop machine learning models to improve anomaly prediction\n")
        f.write("- Create a centralized dashboard for real-time anomaly monitoring\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("This comprehensive analysis demonstrates the value of integrating multiple analytical approaches to water leakage detection. By combining leak detection, consumption pattern analysis, early warning systems, and anomaly detection, we can develop a robust framework for identifying and preventing water leaks while optimizing resource management.\n\n")
        f.write("The results show that with proper data analysis and monitoring, it is possible to detect leaks early, predict consumption patterns, and identify anomalies that may indicate system issues before they become critical problems.\n\n")
        
    print(f"Report generated successfully: {report_file}")


def run_analysis(data_path, output_dir, report_dir):
    """
    Run the comprehensive analysis.
    
    Args:
        data_path (str): Path to the data files
        output_dir (str): Directory to save analysis results
        report_dir (str): Directory to save the report
    """
    print("=== Water Leakage Priority Applications Analysis ===")
    
    # Setup
    setup_directories(data_path, output_dir, report_dir)
    
    # Load and prepare data
    df, sensor_ids, loader = load_and_prepare_data(data_path)
    
    # 1. Leak Detection Analysis
    leak_results = analyze_leak_detection(df, sensor_ids, output_dir)
    
    # 2. Water Consumption Pattern Analysis
    consumption_results = analyze_consumption_patterns(df, sensor_ids, output_dir)
    
    # 3. Early Warning System Analysis
    warning_results = analyze_early_warning(df, sensor_ids, leak_results, output_dir)
    
    # 4. Anomaly Detection Analysis
    anomaly_results = analyze_anomalies(df, sensor_ids, output_dir)
    
    # Generate comprehensive report
    generate_report(leak_results, consumption_results, warning_results, anomaly_results, sensor_ids, report_dir, output_dir)
    
    print("\n=== Analysis Complete ===")
    
    return {
        'leak_results': leak_results,
        'consumption_results': consumption_results,
        'warning_results': warning_results,
        'anomaly_results': anomaly_results
    } 