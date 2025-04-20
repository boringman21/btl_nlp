#!/bin/bash

echo "===================================="
echo "  Setting up environment for btl_nlp"
echo "===================================="
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Please install Python3 before continuing."
    exit 1
fi

echo "Starting virtual environment setup..."
echo

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating new virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo
echo "Installing required packages..."
pip install -r requirements.txt

# Install btl_nlp package in development mode
echo
echo "Installing btl_nlp package..."
pip install -e .

echo
echo "Checking installation..."
python -c "import btl_nlp; print('btl_nlp version:', btl_nlp.__version__)"

echo
echo "========================================="
echo "  Setup complete!"
echo "  - Virtual environment is activated"
echo "  - btl_nlp package is installed"
echo
echo "  To run sample analysis:"
echo "  python run_sample.py"
echo "=========================================" 