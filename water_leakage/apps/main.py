#!/usr/bin/env python3
"""
Main execution functions for water leakage priority applications.
"""

import os
import argparse
import json
import numpy as np
from water_leakage.apps.priority_applications import run_analysis
from water_leakage.apps.visualize import visualize_results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Water Leakage Priority Applications Analysis"
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
        default="./analysis_results",
        help="Directory to save analysis results"
    )
    
    parser.add_argument(
        "--report_dir", 
        type=str, 
        default="./report",
        help="Directory to save the analysis report"
    )
    
    parser.add_argument(
        "--visualization", 
        action="store_true",
        help="Generate visualizations and dashboards"
    )
    
    parser.add_argument(
        "--save_results", 
        action="store_true",
        help="Save analysis results to file for later visualization"
    )
    
    parser.add_argument(
        "--sensor_id", 
        type=str,
        help="Specific sensor ID to focus on in visualizations"
    )
    
    return parser.parse_args()

def save_results(results, output_dir):
    """Save analysis results to files for later use."""
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save as JSON for human readability
    json_path = os.path.join(results_dir, 'analysis_results.json')
    # Create a serializable version of the results (removing non-serializable objects)
    serializable_results = {}
    
    for key, value in results.items():
        if key == 'leak_results':
            # Handle leak events which might contain DataFrame objects
            leak_copy = value.copy()
            if 'leak_events' in leak_copy:
                # Convert DataFrames to dict representations
                serialized_events = {}
                for sensor_id, events_df in leak_copy['leak_events'].items():
                    if hasattr(events_df, 'to_dict'):
                        serialized_events[sensor_id] = {
                            'count': len(events_df),
                            'first_event': events_df.iloc[0].to_dict() if len(events_df) > 0 else {},
                            'last_event': events_df.iloc[-1].to_dict() if len(events_df) > 0 else {}
                        }
                    else:
                        serialized_events[sensor_id] = events_df
                leak_copy['leak_events'] = serialized_events
            serializable_results[key] = leak_copy
        else:
            serializable_results[key] = value
    
    with open(json_path, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    # Save as NPZ for more efficient data storage
    np_path = os.path.join(results_dir, 'analysis_results.npz')
    np.savez_compressed(np_path, **serializable_results)
    
    print(f"Results saved to:\n- {json_path}\n- {np_path}")
    
    return json_path, np_path

def main():
    """Main function to run the water leakage priority applications analysis."""
    # Parse command line arguments
    args = parse_args()
    
    # Create directories if they don't exist
    for directory in [args.output_dir, args.report_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Run the analysis
    print(f"Starting water leakage analysis with data from: {args.data_path}")
    print(f"Results will be saved to: {args.output_dir}")
    print(f"Report will be saved to: {args.report_dir}")
    
    # Run the analysis
    results = run_analysis(
        data_path=args.data_path,
        output_dir=args.output_dir,
        report_dir=args.report_dir
    )
    
    # Save results if requested
    if args.save_results:
        json_path, np_path = save_results(results, args.output_dir)
    
    # Generate visualizations if requested
    if args.visualization:
        visualize_results(results, args.output_dir, args.sensor_id)
    
    print("\n=== Analysis Complete ===")
    print(f"- Total potential leaks detected: {results['leak_results'].get('total_leaks', 0)}")
    print(f"- Report saved to: {args.report_dir}")
    print(f"- Visualizations saved to: {args.output_dir}/dashboard" if args.visualization else "")
    
    return 0

if __name__ == "__main__":
    main() 