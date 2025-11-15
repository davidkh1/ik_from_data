"""
Memory monitoring utilities for tracking RAM usage during processing.

This module provides functions to monitor system and process memory usage,
which is critical for preventing OOM (Out of Memory) kills during heavy
data processing tasks like video frame loading and tracking.
"""

import psutil
import os


def print_memory_usage(location=""):
    """
    Print current memory usage for monitoring.

    Args:
        location: String describing where this is being called from
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem = psutil.virtual_memory()

    process_gb = mem_info.rss / 1024**3
    total_gb = mem.total / 1024**3
    available_gb = mem.available / 1024**3
    used_percent = mem.percent

    print(f"\n[MEMORY] {location}")
    print(f"  Process RAM: {process_gb:.2f} GB")
    print(f"  System: {used_percent:.1f}% used ({available_gb:.2f} GB free / {total_gb:.2f} GB total)")

    # Warning thresholds
    if used_percent > 80:
        print(f"  WARNING: System memory usage > 80%! Risk of systemd-oomd kill!")
    elif used_percent > 70:
        print(f"  CAUTION: System memory usage > 70%")
    elif used_percent > 50:
        print(f"  INFO: System memory usage > 50% (systemd-oomd threshold)")

    return used_percent


def get_memory_info():
    """
    Get current memory information as a dictionary.

    Returns:
        dict: Memory statistics including process and system usage
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem = psutil.virtual_memory()

    return {
        'process_gb': mem_info.rss / 1024**3,
        'system_total_gb': mem.total / 1024**3,
        'system_available_gb': mem.available / 1024**3,
        'system_used_percent': mem.percent
    }
