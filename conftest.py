"""
Makes the package importable when running pytest from the repository root,
without needing to `pip install -e .` first.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)                       # datasets
sys.path.insert(0, os.path.join(ROOT, 'src'))  # si
