# Water Leakage - Leak Detection System

A Python package for analyzing leak detection data from water systems.

## Installation

Clone the repository and install the package:

```bash
git clone https://github.com/boringman21/water_leakage.git
cd water_leakage
pip install -e .
```

Or install the dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
water_leakage/
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

## Usage

### Loading and Transforming Data

```python
from water_leakage.data.data_loader import DataLoader
from water_leakage.data.data_transform import transform_df, add_derived_metrics

# Initialize the data loader
data_loader = DataLoader('/path/to/data_files')

# Load all data
data = data_loader.load_all_data()

# Transform the data
transformed_df = transform_df(data['merged_data'])

# Add derived metrics
final_df = add_derived_metrics(transformed_df)
```

### Visualizing Data

```python
from water_leakage.utils.visualization import plot_flow_and_pressure, plot_correlation

# Plot flow and pressure data for a specific sensor
plot_flow_and_pressure(final_df, sensor_id='8401210607558', save_dir='output/plots')

# Plot correlation matrix
plot_correlation(final_df, sensor_id='8401210607558', save_dir='output/plots')
```

### Fourier Analysis

```python
from water_leakage.models.fourier import plot_fourier_approximation

# Plot Fourier approximation for a specific sensor and metric
plot_fourier_approximation(
    final_df['Timestamp'],
    final_df['8401210607558_Flow'].values,
    label='Flow',
    sensor_id='8401210607558',
    save_dir='output/plots',
    num_terms=10
)
```

## Memory Management

```python
from water_leakage.utils.memory_utils import print_memory_usage, clear_memory

# Check memory usage
print_memory_usage()

# Free up memory
clear_memory()
``` 