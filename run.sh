#!/bin/bash
echo "============================================================"
echo " SIF Precursor Detection Engine - Oil India Limited"
echo "============================================================"

# Check Python
python3 --version > /dev/null 2>&1 || { echo "ERROR: Python 3 not found."; exit 1; }

echo "[1/4] Installing Python dependencies..."
pip install -r requirements.txt --quiet

echo "[2/4] Downloading spaCy model..."
python3 -m spacy download en_core_web_sm --quiet 2>/dev/null || true

echo "[3/4] Generating demo dataset..."
python3 -c "import sys; sys.path.insert(0, '.'); from data.synthetic_reports import generate_dataset; df=generate_dataset(500); df.to_csv('data/sample_reports.csv', index=False); print(f'  Generated {len(df)} reports')"

echo "[4/4] Launching dashboard at http://localhost:8501 ..."
streamlit run app/dashboard.py --server.port 8501 --browser.gatherUsageStats false
