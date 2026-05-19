"""
BTC Forecast V4 — Streamlit Cloud entry point.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'versions', 'hybrid_v4'))

from app_v4 import main

if __name__ == '__main__':
    main()
