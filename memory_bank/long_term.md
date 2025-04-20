# Water Leakage Project - Leak Detection System

## Project Goals

The main goals of this project are:

1. Analyze water system data to detect leaks in pipelines
2. Process and transform time series data from multiple sensors
3. Visualize flow and pressure data to identify patterns
4. Perform correlation analysis between different metrics
5. Apply Fourier analysis for signal processing
6. Identify anomalies and predict unusual patterns in water consumption
7. Create a reusable, well-structured Python package for leak detection analysis
8. Implement comprehensive analysis for priority applications: leak detection, consumption pattern prediction, early warning systems, and anomaly detection
9. Generate visual dashboards and reports for easier interpretation of results

## Project Structure

The project has been refactored from a single Jupyter notebook (NewDat.ipynb) into a structured Python package with the following components:

```
btl_nlp/
│
├── water_leakage/             # Main package
│   ├── __init__.py
│   │
│   ├── data/                  # Data storage and management
│   │   ├── __init__.py
│   │   ├── data_loader.py     # Data loading and preprocessing functionality
│   │   └── data_transform.py  # Data transformation functions
│   │
│   ├── utils/                 # Utility functions
│   │   ├── __init__.py
│   │   ├── memory_utils.py    # Memory monitoring utilities
│   │   └── visualization.py   # Data visualization functions
│   │
│   ├── models/                # Analysis and modeling code
│   │   ├── __init__.py
│   │   └── fourier.py         # Fourier approximation functions
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
├── check_data.py              # Script to check dataset integrity
├── run_sample.py              # Sample script to run basic analysis
├── README.md                  # Project documentation
├── VISUALIZATION.md           # Documentation for visualization features
├── requirements.txt           # Project dependencies
└── setup.py                   # Package setup file
```

## Conventions

### Code Style

1. **Modular Structure**: Separate functionality into logical modules and packages
2. **Documentation**: All functions and classes include docstrings with descriptions and parameter details
3. **Type Hints**: Use descriptive variable names and parameter types for clarity
4. **Error Handling**: Include appropriate error checking and user feedback

### Naming Conventions

1. **Files**: Use snake_case for file names (e.g., `data_loader.py`)
2. **Classes**: Use PascalCase for class names (e.g., `DataLoader`)
3. **Functions**: Use snake_case for function names (e.g., `transform_df()`)
4. **Variables**: Use snake_case for variable names (e.g., `sensor_id`)

### Data Processing

1. **Vectorization**: Use NumPy vectorized operations when possible for better performance
2. **Memory Management**: Include memory optimization using garbage collection when needed
3. **Data Structure**: Standardize data format with timestamps as index and sensor metrics as columns

## Key Components

### Data Loading and Processing

The `data` module handles loading and transforming the raw data:

- `DataLoader` class: Loads CSV files and provides methods to extract sensor IDs
- `transform_df()`: Transforms raw data into a time-series format with sensors as columns
- `add_derived_metrics()`: Calculates additional metrics like pressure difference

### Visualization

The `visualization` module provides plotting utilities:

- `plot_flow_and_pressure()`: Creates time series plots for flow and pressure data
- `plot_correlation()`: Generates correlation matrices for sensor metrics

### Analysis Models

The `models` module includes analytical techniques:

- `fourier_approximation()`: Implements Fourier series approximation
- `plot_fourier_approximation()`: Visualizes original signal vs. Fourier approximation

### Priority Applications (apps module)

The `apps` module contains implementations for specific use cases and applications:

- `priority_applications.py`: Comprehensive implementation of the four priority applications
- `main.py`: Command-line interface and execution orchestration
- `visualize.py`: Dashboard generation and visualization functionality

## Priority Applications

The project focuses on four key applications with highest practical value:

1. **Leak Detection**
   - Analyze pressure differences and abnormal flow patterns
   - Identify potential leak locations by correlating sensor data
   - Reduce water loss and maintenance costs through early detection

2. **Water Consumption Pattern Prediction**
   - Analyze historical usage patterns to forecast future demand
   - Identify peak usage times and seasonal variations
   - Optimize water resource management and distribution

3. **Early Warning System**
   - Detect unusual changes before major leaks occur
   - Provide real-time alerts for rapid response
   - Minimize damage through proactive maintenance

4. **Anomaly Detection and Prediction**
   - Identify unusual patterns that deviate from normal behavior
   - Predict potential anomalies before they become visible problems
   - Differentiate between natural variations and actual system issues

## Visualization Features

The project includes comprehensive visualization capabilities for interpreting analysis results:

1. **Integrated Dashboards**
   - Combines results from all four priority applications in a single view
   - Custom dashboards for each sensor with key metrics and charts
   - Summary statistics and KPIs for quick assessment

2. **Comparative Analysis**
   - Cross-sensor comparison charts
   - Ranking of sensors by leak count and anomaly percentage
   - Identification of most problematic areas in the water system

3. **Result Persistence**
   - Option to save analysis results for later visualization
   - Support for JSON and NPZ file formats
   - Standalone visualization script that works with previously saved results

## How to Run the Analysis

### Priority Applications Analysis

1. Run the `analyze_priority_app.py` script:
   ```
   python analyze_priority_app.py
   ```

2. This will:
   - Load and process data from the specified directory
   - Perform the four priority analyses: leak detection, consumption patterns, early warning, and anomaly detection
   - Generate visualizations in the `analysis_results` directory
   - Create a comprehensive report in the `report` directory

### Visualization Options

1. Include visualizations during analysis:
   ```
   python analyze_priority_app.py --visualization
   ```

2. Save results for later visualization:
   ```
   python analyze_priority_app.py --save_results
   ```

3. Visualize previously saved results:
   ```
   python visualize_water_leakage.py --results_file ./analysis_results/results/analysis_results.json
   ```

For more visualization options, refer to the `VISUALIZATION.md` file.

## Future Improvements

1. Enhance anomaly detection algorithms with machine learning models
2. Implement real-time data processing capabilities for immediate alerts
3. Create a web dashboard for visualization and monitoring
4. Add predictive models for consumption forecasting
5. Develop geospatial analysis to better locate leaks
6. Improve integration with IoT sensor networks
7. Implement pattern recognition for recurring issues
8. Add more statistical analysis methods
9. Improve test coverage with more unit tests
10. Add interactive visualization capabilities with web frameworks
