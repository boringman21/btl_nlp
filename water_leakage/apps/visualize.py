#!/usr/bin/env python3
"""
Visualization module for water leakage priority applications.
Provides interactive visualizations and dashboards for analysis results.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def create_dashboard(results: Dict, output_dir: str, sensor_id: Optional[str] = None):
    """
    Create a comprehensive dashboard for the analysis results.
    
    Args:
        results (Dict): Dictionary containing analysis results
        output_dir (str): Directory to save the dashboard
        sensor_id (str, optional): Specific sensor to visualize. If None, choose the first available.
    """
    # Extract results
    leak_results = results.get('leak_results', {})
    consumption_results = results.get('consumption_results', {})
    warning_results = results.get('warning_results', {})
    anomaly_results = results.get('anomaly_results', {})
    
    # Create dashboard directory
    dashboard_dir = os.path.join(output_dir, 'dashboard')
    os.makedirs(dashboard_dir, exist_ok=True)
    
    # Get sensor IDs
    available_sensors = set()
    for result_set in [leak_results.get('sensor_summaries', {}), 
                     consumption_results.get('sensor_stats', {}),
                     warning_results.get('warning_indicators', {}),
                     anomaly_results.get('anomalies', {})]:
        available_sensors.update(result_set.keys())
    
    available_sensors = list(available_sensors)
    
    if not available_sensors:
        print("No sensor data available for visualization")
        return
    
    # Select sensor to visualize
    if sensor_id is None or sensor_id not in available_sensors:
        sensor_id = available_sensors[0]
        print(f"No specific sensor selected or selected sensor not available. Using sensor {sensor_id}")
    
    # Create a main dashboard image
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'Water Leakage Analysis Dashboard - Sensor {sensor_id}', fontsize=20)
    
    # Define grid layout
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Leak detection
    ax2 = fig.add_subplot(gs[0, 1])  # Consumption patterns
    ax3 = fig.add_subplot(gs[1, 0])  # Early warning
    ax4 = fig.add_subplot(gs[1, 1])  # Anomaly detection
    ax5 = fig.add_subplot(gs[2, :])  # Summary metrics
    
    # 1. Leak Detection Visualization
    _plot_leak_detection(ax1, leak_results, sensor_id)
    
    # 2. Consumption Pattern Visualization
    _plot_consumption_pattern(ax2, consumption_results, sensor_id)
    
    # 3. Early Warning Visualization
    _plot_early_warning(ax3, warning_results, sensor_id)
    
    # 4. Anomaly Detection Visualization
    _plot_anomaly_detection(ax4, anomaly_results, sensor_id)
    
    # 5. Summary Metrics
    _plot_summary_metrics(ax5, results, sensor_id)
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    dashboard_file = os.path.join(dashboard_dir, f'dashboard_{sensor_id}.png')
    plt.savefig(dashboard_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Dashboard generated successfully: {dashboard_file}")
    
    # Create additional visualizations
    _create_comparison_chart(results, dashboard_dir, available_sensors)


def _plot_leak_detection(ax, leak_results, sensor_id):
    """Plot leak detection summary on the given axis."""
    ax.set_title('Leak Detection Analysis', fontsize=14)
    
    # Check if we have data for this sensor
    if not leak_results or 'sensor_summaries' not in leak_results or sensor_id not in leak_results['sensor_summaries']:
        ax.text(0.5, 0.5, 'No leak detection data available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
        return
    
    # Get statistics
    summary = leak_results['sensor_summaries'][sensor_id]
    
    # Create a bar for key metrics
    metrics = ['leak_count', 'pressure_diff_mean', 'pressure_diff_std']
    values = [summary.get(m, 0) for m in metrics]
    labels = ['Leak Count', 'Pressure Diff Mean', 'Pressure Diff Std']
    
    # Create bar plot
    bars = ax.bar(labels, values, color=['#ff6666', '#66b3ff', '#99ff99'])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom')
    
    ax.set_ylabel('Value')
    ax.set_ylim(0, max(values) * 1.2)  # Add 20% headroom


def _plot_consumption_pattern(ax, consumption_results, sensor_id):
    """Plot consumption pattern on the given axis."""
    ax.set_title('Water Consumption Pattern', fontsize=14)
    
    # Check if we have data for this sensor
    if (not consumption_results or 'hourly_patterns' not in consumption_results 
            or sensor_id not in consumption_results['hourly_patterns']):
        ax.text(0.5, 0.5, 'No consumption pattern data available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
        return
    
    # Get hourly pattern
    hourly_data = consumption_results['hourly_patterns'][sensor_id]
    
    # Convert to pandas Series for easier plotting
    hours = sorted(hourly_data.keys())
    values = [hourly_data[h] for h in hours]
    
    # Create line plot
    ax.plot(hours, values, marker='o', linewidth=2, markersize=6, color='#3388ff')
    
    # Add peak and minimum markers
    if 'sensor_stats' in consumption_results and sensor_id in consumption_results['sensor_stats']:
        stats = consumption_results['sensor_stats'][sensor_id]
        peak_hour = stats.get('peak_hour')
        min_hour = stats.get('min_hour')
        
        if peak_hour in hourly_data:
            peak_value = hourly_data[peak_hour]
            ax.plot(peak_hour, peak_value, 'ro', markersize=10, label='Peak')
            ax.text(peak_hour, peak_value * 1.05, f'Peak: {peak_value:.2f}', ha='center')
        
        if min_hour in hourly_data:
            min_value = hourly_data[min_hour]
            ax.plot(min_hour, min_value, 'go', markersize=10, label='Minimum')
            ax.text(min_hour, min_value * 0.95, f'Min: {min_value:.2f}', ha='center')
    
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Flow')
    ax.set_xticks(range(0, 24, 2))  # Show every 2 hours
    ax.grid(True, linestyle='--', alpha=0.7)


def _plot_early_warning(ax, warning_results, sensor_id):
    """Plot early warning system results on the given axis."""
    ax.set_title('Early Warning System', fontsize=14)
    
    # Check if we have data for this sensor
    if (not warning_results or 'warning_indicators' not in warning_results 
            or sensor_id not in warning_results['warning_indicators']):
        ax.text(0.5, 0.5, 'No early warning data available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
        return
    
    # Get warning indicators for different window sizes
    indicators = warning_results['warning_indicators'][sensor_id]
    
    # Create grouped bar chart for different window sizes
    window_sizes = sorted(indicators.keys())
    x = np.arange(len(window_sizes))
    width = 0.35
    
    # Extract warning counts and percentages
    warning_counts = [indicators[w]['warning_count'] for w in window_sizes]
    warning_percentages = [indicators[w]['warning_percentage'] for w in window_sizes]
    
    # Create bars
    ax.bar(x - width/2, warning_counts, width, label='Warning Count', color='#ff9999')
    ax.bar(x + width/2, warning_percentages, width, label='Warning %', color='#99ccff')
    
    # Add labels
    ax.set_xlabel('Window Size')
    ax.set_ylabel('Count / Percentage')
    ax.set_xticks(x)
    ax.set_xticklabels(window_sizes)
    ax.legend()
    
    # Add detection rate if available
    if 'sensitivity_analysis' in warning_results and sensor_id in warning_results['sensitivity_analysis']:
        sensitivity = warning_results['sensitivity_analysis'][sensor_id]
        if sensitivity:
            # Add text about detection rates
            detection_rates = [f"Window {w}: {sensitivity[w]['detection_rate']:.1f}%" 
                             for w in window_sizes if w in sensitivity]
            if detection_rates:
                ax.text(0.5, 0.05, 'Detection rates:\n' + '\n'.join(detection_rates),
                        transform=ax.transAxes, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def _plot_anomaly_detection(ax, anomaly_results, sensor_id):
    """Plot anomaly detection results on the given axis."""
    ax.set_title('Anomaly Detection', fontsize=14)
    
    # Check if we have data for this sensor
    if not anomaly_results or 'anomalies' not in anomaly_results or sensor_id not in anomaly_results['anomalies']:
        ax.text(0.5, 0.5, 'No anomaly detection data available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
        return
    
    # Get anomaly statistics
    anomaly_stats = anomaly_results['anomalies'][sensor_id]
    
    # Create a pie chart showing anomalies vs normal data
    labels = ['Normal Data', 'Anomalies']
    anomaly_count = anomaly_stats.get('count', 0)
    anomaly_pct = anomaly_stats.get('percentage', 0)
    normal_pct = 100 - anomaly_pct
    
    sizes = [normal_pct, anomaly_pct]
    colors = ['#66b3ff', '#ff6666']
    explode = (0, 0.1)  # Explode the anomaly slice
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures circle is drawn
    
    # Add text with additional information
    ax.text(0.5, -0.1, f'Total anomalies: {anomaly_count}', 
            transform=ax.transAxes, ha='center')
    
    if anomaly_stats.get('min_value') is not None and anomaly_stats.get('max_value') is not None:
        ax.text(0.5, -0.2, f"Range: {anomaly_stats['min_value']:.2f} to {anomaly_stats['max_value']:.2f}",
                transform=ax.transAxes, ha='center')


def _plot_summary_metrics(ax, results, sensor_id):
    """Plot summary metrics on the given axis."""
    ax.set_title('Summary Metrics and KPIs', fontsize=14)
    
    # Define metrics to show from each analysis
    metrics = []
    
    # Leak detection metrics
    leak_results = results.get('leak_results', {})
    if 'sensor_summaries' in leak_results and sensor_id in leak_results['sensor_summaries']:
        summary = leak_results['sensor_summaries'][sensor_id]
        if 'leak_count' in summary:
            metrics.append(('Potential Leaks', summary['leak_count']))
        if 'pressure_diff_mean' in summary:
            metrics.append(('Avg Pressure Diff', f"{summary['pressure_diff_mean']:.2f}"))
    
    # Consumption pattern metrics
    consumption_results = results.get('consumption_results', {})
    if 'sensor_stats' in consumption_results and sensor_id in consumption_results['sensor_stats']:
        stats = consumption_results['sensor_stats'][sensor_id]
        if 'peak_hour' in stats and 'peak_value' in stats:
            metrics.append(('Peak Usage Hour', f"{stats['peak_hour']} ({stats['peak_value']:.2f})"))
        if 'variance' in stats:
            metrics.append(('Flow Variance', f"{stats['variance']:.2f}"))
    
    # Early warning metrics
    warning_results = results.get('warning_results', {})
    if 'warning_indicators' in warning_results and sensor_id in warning_results['warning_indicators']:
        indicators = warning_results['warning_indicators'][sensor_id]
        # Get middle window size
        window_sizes = sorted(indicators.keys())
        if window_sizes:
            middle_window = window_sizes[len(window_sizes)//2]
            metrics.append(('Warning Count', indicators[middle_window]['warning_count']))
            metrics.append(('Warning %', f"{indicators[middle_window]['warning_percentage']:.2f}%"))
            
            # Add detection rate if available
            if ('sensitivity_analysis' in warning_results and 
                sensor_id in warning_results['sensitivity_analysis'] and
                middle_window in warning_results['sensitivity_analysis'][sensor_id]):
                detection_rate = warning_results['sensitivity_analysis'][sensor_id][middle_window]['detection_rate']
                metrics.append(('Detection Rate', f"{detection_rate:.2f}%"))
    
    # Anomaly detection metrics
    anomaly_results = results.get('anomaly_results', {})
    if 'anomalies' in anomaly_results and sensor_id in anomaly_results['anomalies']:
        anomaly_stats = anomaly_results['anomalies'][sensor_id]
        metrics.append(('Anomalies', anomaly_stats.get('count', 0)))
        metrics.append(('Anomaly %', f"{anomaly_stats.get('percentage', 0):.2f}%"))
    
    # Create a table with metrics
    if metrics:
        # Create the table
        cell_text = []
        cell_colors = []
        for i, (label, value) in enumerate(metrics):
            cell_text.append([label, value])
            # Alternate row colors
            if i % 2 == 0:
                cell_colors.append(['#f2f2f2', '#f2f2f2'])
            else:
                cell_colors.append(['#e6e6e6', '#e6e6e6'])
        
        the_table = ax.table(cellText=cell_text,
                            colLabels=['Metric', 'Value'],
                            colWidths=[0.5, 0.3],
                            cellColours=cell_colors,
                            loc='center')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(12)
        the_table.scale(1, 1.5)
        ax.axis('tight')
        ax.axis('off')
    else:
        ax.text(0.5, 0.5, 'No summary metrics available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)


def _create_comparison_chart(results, dashboard_dir, sensor_ids):
    """Create a comparison chart across multiple sensors."""
    if not sensor_ids or len(sensor_ids) < 2:
        return  # Not enough sensors for comparison
    
    # Create a figure for comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Sensor Comparison', fontsize=18)
    
    # 1. Compare leak counts
    leak_results = results.get('leak_results', {})
    leak_counts = []
    for sensor_id in sensor_ids:
        if ('sensor_summaries' in leak_results and sensor_id in leak_results['sensor_summaries'] and
            'leak_count' in leak_results['sensor_summaries'][sensor_id]):
            leak_counts.append((sensor_id, leak_results['sensor_summaries'][sensor_id]['leak_count']))
    
    # Sort by leak count in descending order
    leak_counts.sort(key=lambda x: x[1], reverse=True)
    
    # Plot top 10 sensors by leak count
    top_leak_sensors = leak_counts[:10]
    if top_leak_sensors:
        sensor_ids_leak = [s[0] for s in top_leak_sensors]
        leak_values = [s[1] for s in top_leak_sensors]
        
        # Create horizontal bar chart
        ax1.barh(sensor_ids_leak, leak_values, color='#ff6666')
        ax1.set_title('Top 10 Sensors by Leak Count', fontsize=14)
        ax1.set_xlabel('Number of Potential Leaks')
        ax1.set_ylabel('Sensor ID')
        ax1.invert_yaxis()  # To have the highest value on top
    else:
        ax1.text(0.5, 0.5, 'No leak data available for comparison', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax1.transAxes, fontsize=12)
    
    # 2. Compare anomaly percentages
    anomaly_results = results.get('anomaly_results', {})
    anomaly_pcts = []
    for sensor_id in sensor_ids:
        if ('anomalies' in anomaly_results and sensor_id in anomaly_results['anomalies'] and
            'percentage' in anomaly_results['anomalies'][sensor_id]):
            anomaly_pcts.append((sensor_id, anomaly_results['anomalies'][sensor_id]['percentage']))
    
    # Sort by anomaly percentage in descending order
    anomaly_pcts.sort(key=lambda x: x[1], reverse=True)
    
    # Plot top 10 sensors by anomaly percentage
    top_anomaly_sensors = anomaly_pcts[:10]
    if top_anomaly_sensors:
        sensor_ids_anomaly = [s[0] for s in top_anomaly_sensors]
        anomaly_values = [s[1] for s in top_anomaly_sensors]
        
        # Create horizontal bar chart
        ax2.barh(sensor_ids_anomaly, anomaly_values, color='#66b3ff')
        ax2.set_title('Top 10 Sensors by Anomaly Percentage', fontsize=14)
        ax2.set_xlabel('Anomaly Percentage (%)')
        ax2.set_ylabel('Sensor ID')
        ax2.invert_yaxis()  # To have the highest value on top
    else:
        ax2.text(0.5, 0.5, 'No anomaly data available for comparison', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax2.transAxes, fontsize=12)
    
    # Save the comparison chart
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
    comparison_file = os.path.join(dashboard_dir, 'sensor_comparison.png')
    plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Sensor comparison chart generated: {comparison_file}")


def visualize_results(results: Dict, output_dir: str, sensor_id: Optional[str] = None):
    """
    Visualize analysis results from priority applications.
    
    Args:
        results (Dict): Dictionary containing analysis results
        output_dir (str): Directory to save visualizations
        sensor_id (str, optional): Specific sensor to visualize. If None, select first available.
    """
    print("\n=== Generating Visualizations ===")
    
    # Create comprehensive dashboard
    create_dashboard(results, output_dir, sensor_id)
    
    print("All visualizations completed successfully")


def main(results_file: str, output_dir: str, sensor_id: Optional[str] = None):
    """
    Main function to generate visualizations from saved results.
    
    Args:
        results_file (str): Path to the saved results (.npz or .json)
        output_dir (str): Directory to save visualizations
        sensor_id (str, optional): Specific sensor to visualize
    """
    # Load results
    if results_file.endswith('.npz'):
        results = dict(np.load(results_file, allow_pickle=True))
    elif results_file.endswith('.json'):
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        raise ValueError("Unsupported results file format. Use .npz or .json")
    
    # Generate visualizations
    visualize_results(results, output_dir, sensor_id)
    
    return 0


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Visualize water leakage analysis results")
    parser.add_argument("results_file", help="Path to the saved results file (.npz or .json)")
    parser.add_argument("--output_dir", default="./visualization_output", help="Directory to save visualizations")
    parser.add_argument("--sensor_id", help="Specific sensor ID to visualize")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args.results_file, args.output_dir, args.sensor_id) 