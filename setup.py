from setuptools import setup, find_packages

setup(
    name="water_leakage",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "psutil>=5.8.0",
        "scikit-learn>=1.0.0",
    ],
    author="Water Leakage Team",
    author_email="example@example.com",
    description="A package for processing and analyzing leak detection data.",
    keywords="water, leak detection, data analysis",
    python_requires=">=3.8",
) 