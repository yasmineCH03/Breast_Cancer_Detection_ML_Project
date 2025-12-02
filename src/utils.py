"""
Utility functions for breast cancer detection project.
"""

import os
import json


def save_model(model, filepath):
    """Save trained model to file."""
    pass


def load_model(filepath):
    """Load trained model from file."""
    pass


def save_results(results, filepath):
    """Save results to file."""
    pass


def create_directory(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
