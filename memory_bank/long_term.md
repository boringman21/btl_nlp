# BTL NLP Project - Leak Detection System

## Project Goals

The main goals of this project are:

1. Analyze water system data to detect leaks in pipelines
2. Process and transform time series data from multiple sensors
3. Visualize flow and pressure data to identify patterns
4. Perform correlation analysis between different metrics
5. Apply Fourier analysis for signal processing
6. Create a reusable, well-structured Python package for leak detection analysis

## Project Structure

The project has been refactored from a single Jupyter notebook (NewDat.ipynb) into a structured Python package with the following components:

```
btl_nlp/
│
├── data/                    # Data storage and management
│   ├── __init__.py
│   ├── data_loader.py       # Data loading and preprocessing functionality
│   └── data_transform.py    # Data transformation functions
│
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── memory_utils.py      # Memory monitoring utilities
│   └── visualization.py     # Data visualization functions
│
├── models/                  # Analysis and modeling code
│   ├── __init__.py
│   └── fourier.py           # Fourier approximation functions
│
├── tests/                   # Unit tests
│   └── __init__.py
│
├── notebooks/               # Jupyter notebooks
│   └── analysis.ipynb       # Original analysis converted to use the package
│
├── README.md                # Project documentation
├── requirements.txt         # Project dependencies
└── setup.py                 # Package setup file
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

## Usage Examples

```python
# Load and transform data
from btl_nlp.data.data_loader import DataLoader
from btl_nlp.data.data_transform import transform_df, add_derived_metrics

loader = DataLoader('/path/to/data')
data = loader.load_all_data()
result_df = transform_df(data['merged_data'])
result_df = add_derived_metrics(result_df)

# Visualize sensor data
from btl_nlp.utils.visualization import plot_flow_and_pressure
plot_flow_and_pressure(result_df, sensor_id='8401210607558')

# Memory management
from btl_nlp.utils.memory_utils import clear_memory
clear_memory()
```

## Future Improvements

1. Add anomaly detection algorithms for leak identification
2. Implement real-time data processing capabilities
3. Create a web dashboard for visualization
4. Add more statistical analysis methods
5. Improve test coverage with more unit tests
