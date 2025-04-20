#!/usr/bin/env python3
"""
Standalone script for visualizing water leakage analysis results.
Use this script to create visualizations and dashboards from previously saved analysis results.
"""

import os
import argparse
import json
import numpy as np
from water_leakage.apps.visualize import visualize_results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize water leakage analysis results from saved data"
    )
    
    parser.add_argument(
        "--results_file", 
        type=str, 
        default="./analysis_results/results/analysis_results.json",
        help="Path to the saved analysis results file (.json or .npz)"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./visualization_output",
        help="Directory to save visualizations"
    )
    
    parser.add_argument(
        "--sensor_id", 
        type=str,
        help="Specific sensor ID to focus on in visualizations"
    )
    
    return parser.parse_args()

def load_results(results_file):
    """Load analysis results from saved file."""
    if results_file.endswith('.npz'):
        # Load from NumPy archive
        try:
            results = dict(np.load(results_file, allow_pickle=True))
            print(f"Results loaded from NPZ file: {results_file}")
            return results
        except Exception as e:
            print(f"Error loading NPZ file: {e}")
            return None
    elif results_file.endswith('.json'):
        # Load from JSON file
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            print(f"Results loaded from JSON file: {results_file}")
            return results
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return None
    else:
        print(f"Unsupported file format: {results_file}")
        print("Please provide a .json or .npz file")
        return None

def main():
    """Main function to run the visualization."""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load saved results
    results = load_results(args.results_file)
    if not results:
        print("Failed to load results. Exiting.")
        return 1
    
    # Generate visualizations
    visualize_results(results, args.output_dir, args.sensor_id)
    
    print(f"\nVisualizations created successfully in: {args.output_dir}/dashboard")
    return 0

if __name__ == "__main__":
    exit(main()) 