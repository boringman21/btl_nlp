#!/usr/bin/env python3
"""
Unit tests for visualization module
"""

import unittest
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from water_leakage.utils.visualization import (
    plot_flow_and_pressure,
    plot_correlation,
    plot_all_sensors,
    plot_predictions
)

class TestVisualization(unittest.TestCase):
    """Test cases for visualization functions"""
    
    def setUp(self):
        """Set up test data"""
        # Create a simple DataFrame for testing
        dates = pd.date_range(start='2023-01-01', periods=100, freq='15min')
        
        # Generate test data
        self.test_df = pd.DataFrame({
            'Timestamp': dates,
            'Test001_Flow': np.sin(np.linspace(0, 4*np.pi, 100)) + 10,
            'Test001_Pressure_1': np.cos(np.linspace(0, 4*np.pi, 100)) + 5,
            'Test001_Pressure_2': np.cos(np.linspace(0, 4*np.pi, 100)) + 8,
            'Test001_Pressure_Diff': np.cos(np.linspace(0, 4*np.pi, 100)) - 3,
            'Test002_Flow': np.sin(np.linspace(0, 4*np.pi, 100)) + 12,
            'Test002_Pressure_1': np.cos(np.linspace(0, 4*np.pi, 100)) + 6,
            'Test002_Pressure_2': np.cos(np.linspace(0, 4*np.pi, 100)) + 9,
            'Test002_Pressure_Diff': np.cos(np.linspace(0, 4*np.pi, 100)) - 3,
        })
        
        # Create a temporary directory for test outputs
        self.test_output_dir = os.path.join(os.path.dirname(__file__), 'test_output')
        os.makedirs(self.test_output_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up after tests"""
        # Close all plots
        plt.close('all')
        
        # Optionally, remove test output files
        for file in os.listdir(self.test_output_dir):
            if file.endswith('.png'):
                os.remove(os.path.join(self.test_output_dir, file))
    
    def test_plot_flow_and_pressure(self):
        """Test flow and pressure plotting function"""
        # Test with first sensor
        fig = plot_flow_and_pressure(self.test_df, 'Test001', save_dir=self.test_output_dir)
        self.assertIsNotNone(fig)
        
        # Check that the file was created
        expected_file = os.path.join(self.test_output_dir, 'Test001_flow_pressure.png')
        self.assertTrue(os.path.exists(expected_file))
    
    def test_plot_correlation(self):
        """Test correlation matrix plotting function"""
        # Test with first sensor
        fig = plot_correlation(self.test_df, 'Test001', save_dir=self.test_output_dir)
        self.assertIsNotNone(fig)
        
        # Check that the file was created
        expected_file = os.path.join(self.test_output_dir, 'Test001_correlation.png')
        self.assertTrue(os.path.exists(expected_file))
    
    def test_plot_all_sensors(self):
        """Test plotting for all sensors"""
        # Test plotting flow for all sensors
        fig = plot_all_sensors(self.test_df, metric='Flow', save_dir=self.test_output_dir)
        self.assertIsNotNone(fig)
        
        # Check that the file was created
        expected_file = os.path.join(self.test_output_dir, 'all_sensors_Flow.png')
        self.assertTrue(os.path.exists(expected_file))
    
    def test_plot_predictions(self):
        """Test prediction plotting function"""
        # Create prediction data
        prediction = {
            'Flow': 11.5,
            'Pressure_1': 5.5,
            'Pressure_2': 8.5
        }
        
        # Next timestamp
        next_timestamp = self.test_df['Timestamp'].max() + timedelta(minutes=15)
        
        # Test plotting predictions
        fig = plot_predictions(
            self.test_df,
            prediction,
            'Test001',
            next_timestamp,
            os.path.join(self.test_output_dir, 'test_prediction.png')
        )
        
        self.assertIsNotNone(fig)
        
        # Check that the file was created
        expected_file = os.path.join(self.test_output_dir, 'test_prediction.png')
        self.assertTrue(os.path.exists(expected_file))

if __name__ == '__main__':
    unittest.main() 