# Installation and Usage Guide for btl_nlp

This guide will help you install and run the `btl_nlp` package using a Python virtual environment.

## 1. System Requirements

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, if you want to clone the repository)

## 2. Create Virtual Environment

### Windows

```bash
# Navigate to the project directory
cd path\to\btl_nlp

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

### macOS/Linux

```bash
# Navigate to the project directory
cd path/to/btl_nlp

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

## 3. Install Package and Dependencies

There are two ways to install the `btl_nlp` package:

### Method 1: Install from requirements.txt

```bash
# Make sure your virtual environment is activated
pip install -r requirements.txt
```

### Method 2: Install package in development mode

```bash
# Make sure your virtual environment is activated
pip install -e .
```

This command will install the `btl_nlp` package in development mode, allowing you to modify the source code without reinstalling.

## 4. Verify Installation

To verify that the package has been installed successfully, you can run the following Python code:

```bash
python -c "import btl_nlp; print(btl_nlp.__version__)"
```

If it displays a version number (e.g., 0.1.0), the installation was successful.

## 5. Running the Code

### Using Jupyter Notebook/Lab

```bash
# Activate virtual environment (if not already activated)
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Install Jupyter (if not installed)
pip install jupyterlab

# Run Jupyter Lab
jupyter lab
```

Then, open an existing notebook or create a new one to use the `btl_nlp` package:

```python
# Import required modules
from btl_nlp.data.data_loader import DataLoader
from btl_nlp.data.data_transform import transform_df, add_derived_metrics
from btl_nlp.utils.visualization import plot_flow_and_pressure, plot_correlation

# Initialize DataLoader with path to data files
data_loader = DataLoader('/path/to/data_files')

# Load data
data = data_loader.load_all_data()

# Transform data
transformed_df = transform_df(data['merged_data'])

# Add derived metrics
final_df = add_derived_metrics(transformed_df)

# Create plots
plot_flow_and_pressure(final_df, sensor_id='8401210607558')
```

### Using Python Script

Create a Python file (e.g., `run_analysis.py`) and add your code:

```python
# run_analysis.py
from btl_nlp.data.data_loader import DataLoader
from btl_nlp.data.data_transform import transform_df
from btl_nlp.utils.visualization import plot_flow_and_pressure
import matplotlib.pyplot as plt

# Perform data analysis
data_loader = DataLoader('/path/to/data_files')
data = data_loader.load_all_data()
result_df = transform_df(data['merged_data'])

# Create and save plots
fig = plot_flow_and_pressure(result_df, sensor_id='8401210607558', save_dir='output_plots')
plt.show()
```

Then run the script:

```bash
python run_analysis.py
```

## 6. Deactivate Virtual Environment

When you're done working, you can deactivate the virtual environment:

### Windows/macOS/Linux

```bash
deactivate
```

---

## Troubleshooting Common Issues

### ImportError: No module named 'btl_nlp'

If you encounter this error, ensure:
1. Your virtual environment is activated
2. The package is installed correctly
3. You're running Python from the correct virtual environment

### ModuleNotFoundError: No module named 'numpy' (or other libraries)

Make sure you have installed all dependencies:

```bash
pip install -r requirements.txt
```

### Data Reading Errors

Ensure the data path is correct and you have access permissions to the data files. 