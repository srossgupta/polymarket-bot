#!/usr/bin/env python3
"""Convenience script to run the Polymarket bot."""
import sys
import os

# Add parent dir to path so the package can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from polymarket_bot.cli import main

if __name__ == "__main__":
    main()
