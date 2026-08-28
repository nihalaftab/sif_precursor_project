"""
Streamlit Cloud root entrypoint — SIF Precursor Detection Engine
Oil India Limited | HSSE AI/NLP Engine

This file is required at the root for Streamlit Cloud Community deployment.
It mirrors app/dashboard.py content with correct path setup.
"""
import sys
import os

# Ensure project root is in path so engine/utils/data modules import correctly
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# Execute the dashboard — exec() preserves the Streamlit top-level context
# which is required for st.set_page_config() and other top-level calls.
# ─────────────────────────────────────────────────────────────────────────────
_dashboard_path = os.path.join(ROOT, "app", "dashboard.py")
with open(_dashboard_path, "r", encoding="utf-8") as _f:
    exec(compile(_f.read(), _dashboard_path, "exec"), {"__name__": "__main__", "__file__": _dashboard_path})
