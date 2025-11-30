"""
Components package for Student Performance Prediction System
"""

# This file makes the components directory a Python package
# It can be empty or contain package-level imports

__version__ = "1.0.0"
__author__ = "Student Performance Prediction Team"

# Optional: Import all component modules for easier access
from . import charts
from . import forms
from . import cards

__all__ = ['charts', 'forms', 'cards']