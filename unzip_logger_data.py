#!/usr/bin/env python3
"""
Logger Data Extraction Utility

This script extracts data from the Logger Data zip file to a specified directory.
"""

import os
import sys
import zipfile
from pathlib import Path


def extract_zip(zip_path: str, target_dir: str) -> bool:
    """
    Extract a zip file to the target directory.
    
    Args:
        zip_path: Path to the zip file
        target_dir: Directory to extract the contents to
        
    Returns:
        bool: True if extraction was successful, False otherwise
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        return True
    except Exception as e:
        print(f"Extraction error: {str(e)}")
        return False


def main():
    """Main function to handle the extraction process."""
    # Default parameters
    ZIP_FILE = "Logger_Data_2024_Bau_Bang-2.zip"
    OUTPUT_DIR = "LeakDataset"
    
    # Validate zip file exists
    zip_path = Path(ZIP_FILE)
    if not zip_path.exists():
        print(f"Error: Zip file '{ZIP_FILE}' not found.")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    os.makedirs(output_path, exist_ok=True)
    
    # Extract files
    print(f"Extracting data from '{zip_path}' to '{output_path}'...")
    
    if extract_zip(zip_path, output_path):
        print(f"✓ Extraction completed successfully!")
        print(f"✓ Files available at: {os.path.abspath(output_path)}")
    else:
        print("✗ Extraction failed.")
        sys.exit(1)


if __name__ == "__main__":
    main() 