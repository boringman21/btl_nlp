# Water Leakage Analysis System

A comprehensive Python package for analyzing water system data to detect leaks and predict consumption patterns.

## Features

This package provides tools for:

1. **Leak Detection**: Analyze pressure differences and flow patterns to identify potential leaks
2. **Consumption Pattern Analysis**: Visualize and forecast water usage patterns
3. **Early Warning System**: Detect signs of potential leaks before they become severe
4. **Anomaly Detection**: Identify unusual patterns in water system data
5. **Time Series Prediction**: Predict the next 15-minute tick based on 6 hours of historical data

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/water-leakage.git
cd water-leakage

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Analysis

```bash
# Run priority applications analysis
python analyze_priority_app.py --data_path ./LeakDataset/Logger_Data_2024_Bau_Bang-2
```

### Visualization

```bash
# Run analysis with visualization
python analyze_priority_app.py --visualization

# Visualize previously saved results
python visualize_water_leakage.py --results_file ./analysis_results/results/analysis_results.json
```

### Time Series Prediction

```bash
# Predict the next 15-minute tick for all sensors
python predict_next_tick.py --data_path ./LeakDataset/Logger_Data_2024_Bau_Bang-2 --visualize

# Predict for a specific sensor
python predict_next_tick.py --sensor_id 8401210607558 --visualize
```

## Module Structure

```
btl_nlp/
│
├── water_leakage/             # Main package
│   ├── __init__.py
│   │
│   ├── data/                  # Data storage and management
│   │   ├── __init__.py
│   │   ├── data_loader.py     # Data loading functionality
│   │   └── data_transform.py  # Data transformation functions
│   │
│   ├── utils/                 # Utility functions
│   │   ├── __init__.py
│   │   ├── memory_utils.py    # Memory monitoring utilities
│   │   └── visualization.py   # Data visualization functions
│   │
│   ├── models/                # Analysis and modeling code
│   │   ├── __init__.py
│   │   ├── fourier.py         # Fourier approximation functions
│   │   └── time_series.py     # Time series prediction (LSTM)
│   │
│   ├── apps/                  # Application-specific implementations
│   │   ├── __init__.py
│   │   ├── priority_applications.py  # Comprehensive analysis for priority applications
│   │   ├── main.py            # Main execution functions with CLI
│   │   └── visualize.py       # Dashboard and visualization functions
│   │
│   ├── tests/                 # Unit tests
│   │   └── __init__.py
│   │
│   └── notebooks/             # Jupyter notebooks
│       └── analysis.ipynb     # Original analysis converted to use the package
│
├── analyze_priority_app.py    # Script to run priority applications analysis
├── visualize_water_leakage.py # Script to create visualizations from saved results
├── predict_next_tick.py       # Script to predict the next 15-minute tick
├── check_data.py              # Script to check dataset integrity
├── run_sample.py              # Sample script to run basic analysis
├── README.md                  # Project documentation
├── requirements.txt           # Project dependencies
└── setup.py                   # Package setup file
```

## Time Series Prediction

The package implements time series prediction using LSTM (Long Short-Term Memory) networks to forecast sensor values for the next 15-minute tick based on 6 hours of historical data.

### How It Works

1. **Data Preprocessing**: Historical data is loaded, normalized, and feature-engineered
2. **Sequence Creation**: Time series data is converted into sliding window sequences (24 steps of 15 minutes each = 6 hours)
3. **Model Training**: A bidirectional LSTM model is trained on the historical data
4. **Prediction**: The trained model predicts the values for the next 15-minute tick

### Model Architecture

The LSTM model uses the following architecture:
- Bidirectional LSTM layer with 64 units
- ReLU activation function
- Dense output layer

### Example

```python
from water_leakage.models.time_series import predict_next_tick

# Predict the next 15-minute tick for a specific sensor
prediction = predict_next_tick(
    df=result_df,
    sensor_id='8401210607558',
    feature_cols=['Flow', 'Pressure_1', 'Pressure_2']
)

print(f"Predicted Flow: {prediction['Flow']:.2f}")
print(f"Predicted Pressure_1: {prediction['Pressure_1']:.2f}")
print(f"Predicted Pressure_2: {prediction['Pressure_2']:.2f}")
```

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow (for LSTM models)
- joblib (for model persistence)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset provided by Bau Bang Industrial Park water management system
- Based on research from the Natural Language Processing department 