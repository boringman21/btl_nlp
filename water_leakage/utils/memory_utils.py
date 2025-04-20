"""
Memory management utilities.
"""

import gc
import psutil
import os

def print_memory_usage():
    """
    Print the current memory usage of the Python process.
    
    Returns:
        float: Memory usage in MB
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    memory_mb = mem_info.rss / (1024 * 1024)
    print(f"Memory Usage: {memory_mb:.2f} MB")
    return memory_mb

def clear_memory():
    """
    Force garbage collection to free up memory.
    
    Returns:
        float: Memory usage after cleanup in MB
    """
    # Print memory usage before cleanup
    before = print_memory_usage()
    
    # Force garbage collection
    gc.collect()
    
    # Print memory usage after cleanup
    after = print_memory_usage()
    
    # Print the difference
    print(f"Memory freed: {before - after:.2f} MB")
    
    return after 