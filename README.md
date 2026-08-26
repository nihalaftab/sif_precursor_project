# 🛢️ SIF Precursor Detection Engine — Oil India Limited

> **AI/NLP engine to automatically classify Serious Injury & Fatality (SIF) precursors in OIL's UA/UC observations, near-miss, and incident reports. Maps each report to IOGP Life-Saving Rules and surfaces recurring risk patterns via an interactive dashboard.**

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
│  CSV / Excel upload  ──►  DuckDB in-process database             │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                       NLP/ML ENGINE                              │
│  Preprocessor ──► SIF Classifier ──► LSR Tagger ──► Patterns    │
│  (spaCy + sentence-transformers)    (IOGP rules)  (BERTopic)     │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                   STREAMLIT DASHBOARD (5 pages)                  │
│  Upload │ Risk Heatmap │ LSR Analysis │ Patterns │ Explorer      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Windows
```bat
cd sif_precursor_engine
run.bat
```

### Manual Setup
```bash
# 1. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download spaCy model
python -m spacy download en_core_web_sm

# 4. Generate demo data
python data/synthetic_reports.py

# 5. Launch dashboard
streamlit run app/dashboard.py
```

Dashboard opens at: **http://localhost:8501**

---

## 📁 Project Structure

```
sif_precursor_engine/
├── data/
│   ├── synthetic_reports.py      # 500-report synthetic OIL dataset generator
│   └── sample_reports.csv        # Pre-generated demo data
│
├── engine/
│   ├── preprocessor.py           # Text cleaning, NER, embeddings
│   ├── sif_classifier.py         # SIF-potential classification
│   ├── lsr_tagger.py             # IOGP Life-Saving Rule tagging
│   └── pattern_miner.py          # Precursor pattern mining (BERTopic)
│
├── app/
│   └── dashboard.py              # Streamlit 5-page interactive dashboard
│
├── utils/
│   ├── config.py                 # Central configuration & keyword lists
│   ├── db.py                     # DuckDB wrapper
│   └── exporter.py               # Excel / PDF export
│
├── tests/
│   └── test_engine.py            # Pytest unit tests (25+ test cases)
│
├── requirements.txt
├── run.bat                       # One-click Windows launcher
└── run.sh                        # Linux/Mac launcher
```

---

## 🧠 How the AI Works

### 1. SIF Classification

Each report is scored using a **keyword + heuristic ensemble**:

| Signal Type | Weight | Examples |
|---|---|---|
| High-energy hazards | 50% | electrical, H2S, pressure, fire, crush |
| Barrier failures | 30% | no lockout, bypass, expired permit, no gas test |
| Near-fatal signals | 20% | "could have been fatal", near miss, critical |

**SIF Score ≥ 0.50 → flagged as SIF-Potential**

Optional: Enable **Zero-Shot LLM** mode (`facebook/bart-large-mnli`) in sidebar for more accurate classification using natural language inference.

### 2. IOGP Life-Saving Rule Tagging

9 rules are mapped using a **two-stage approach**:
1. **Keyword matching** — high recall, oil-field specific terms
2. **Semantic similarity** — cosine similarity vs. rule description embeddings (`all-MiniLM-L6-v2`)

Final tag = highest-scoring rule above threshold (0.30).

### 3. Precursor Pattern Mining

**BERTopic** (or TF-IDF + KMeans fallback) clusters SIF-flagged reports into recurring precursor themes:
- Activity + Location + Barrier failure triads
- Top 10 themes with keyword frequency

---

## 📊 Dashboard Pages

| Page | Description |
|---|---|
| 📤 **Upload & Analyze** | Upload CSV or use 500-report demo; runs full pipeline |
| 🔴 **SIF Risk Heatmap** | Site-level SIF density ranking, bubble chart, cross-heatmap |
| 🏷️ **Life-Saving Rules** | LSR bar chart, treemap (LSR × Site), monthly trend |
| 🔍 **Pattern Explorer** | Precursor pattern table, keyword frequency, entity breakdown |
| 📋 **Report Explorer** | Searchable table, per-report drill-down with SIF score gauge |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Tests cover:
- Abbreviation expansion and text normalization
- Signal scoring (energy, barrier, near-fatal)
- Entity extraction (equipment, location, activity)
- SIF classifier accuracy on obvious SIF/non-SIF cases
- LSR tagger accuracy for all 9 rules
- Synthetic data generator integrity

---

## 📥 Input CSV Format

| Column | Required | Description |
|---|---|---|
| `narrative` | ✅ Yes | Free-text safety report description |
| `report_id` | Optional | Unique identifier |
| `date` | Optional | Report date (YYYY-MM-DD) |
| `site` | Optional | Site/location name |
| `department` | Optional | Department name |
| `report_type` | Optional | UA / UC / Near Miss / Incident |

---

## ⚙️ Configuration

All settings in [`utils/config.py`](utils/config.py):

| Parameter | Default | Description |
|---|---|---|
| `SIF_SCORE_THRESHOLD` | 0.50 | SIF classification cutoff |
| `LSR_SIMILARITY_THRESHOLD` | 0.30 | Min score to tag an LSR rule |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence embedding model |
| `ZERO_SHOT_MODEL` | `facebook/bart-large-mnli` | LLM for zero-shot classification |
| `WEIGHT_LLM` | 0.60 | LLM weight in ensemble |
| `WEIGHT_KEYWORD` | 0.40 | Keyword weight in ensemble |

---

## 🏭 IOGP Life-Saving Rules (9 Rules)

| # | Rule | Key Signals |
|---|---|---|
| 1 | ⚡ Energy Isolation | LOTO, lockout, isolation, zero energy |
| 2 | 🚪 Confined Space | tank entry, H2S, atmospheric test, standby man |
| 3 | 🔥 Hot Work | welding, spark, flammable, gas test, fire watch |
| 4 | 🎯 Line of Fire | dropped object, struck by, exclusion zone |
| 5 | 🪜 Working at Height | fall, scaffold, harness, edge protection |
| 6 | 🏗️ Safe Mechanical Lifting | crane, sling, rigging, lift plan |
| 7 | 📋 Work Authorisation | PTW, permit, JSA, authorization |
| 8 | 🚗 Driving | speeding, seatbelt, fatigue, collision |
| 9 | 🔓 Bypassing Safety Controls | bypass, override, interlock, guard removal |

---

## 📤 Export Formats

- **Excel (.xlsx)** — Full classified dataset with conditional formatting (SIF = red rows)
- **PDF** — Summary report (top 100 records) via ReportLab

---

## 🔮 Future Enhancements

- Fine-tuned BERT model with labeled OIL incident data
- Integration with OIL's HSSE platform API
- Multilingual support (Hindi / Assamese)
- Real-time streaming ingestion
- Email / SMS alert on HIGH-confidence SIF detection
- Docker container for enterprise deployment

---

## 👥 Built For

**Oil India Limited | HSSE Department**  
Problem Statement ID: 26165  
Theme: Smart Automation | Category: Software

---

*Built with Python · Streamlit · sentence-transformers · BERTopic · Plotly · DuckDB*
