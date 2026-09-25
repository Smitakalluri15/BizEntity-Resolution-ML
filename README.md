# BizEntity-Resolution-ML

**High-Performance Multi-Source Business Entity Resolution System**  
*Built for the Amazon ML Challenge 2026 — Business Entity Resolution Track*

---

## 📌 Problem Overview

In large-scale commercial platforms, business identity data arrives from multiple independent sources with noisy, incomplete, and cross-script variations. Given:
- **Source 1**: Reference entity repository (deduplicated).
- **Source 2 & Source 3**: External noisy multi-lingual records (12.5M+ total records).

The objective is to accurately map every Source 1 business entity to its corresponding records across Source 2 and Source 3, optimizing the competition evaluation metric: **Macro $F_{0.5}$** (heavily penalizing false positive mis-merges).

---

## 🏛️ System Architecture

```
                    ┌────────────────────────┐
                    │ Raw Business Records   │
                    │ (Source 1, S2, S3)     │
                    └───────────┬────────────┘
                                │
                                ▼
            ┌────────────────────────────────────────┐
            │ Phase 2 & 2.5: Normalization Pipeline   │
            │  • Script detection & transliteration │
            │  • Legal suffix stripping/transposition│
            │  • Address & PIN standardization       │
            │  • Additive phonetic reduction layer   │
            └───────────────────┬────────────────────┘
                                │
                                ▼
            ┌────────────────────────────────────────┐
            │ Phase 5: Multi-Strategy Blocking       │
            │  • Name token & word-set blocking      │
            │  • Phonetic code inverted index        │
            │  • Geographic & Postal block key       │
            │  • Post-Capped Model Recall: 98.28%    │
            │  • (Pre-Cap Theoretical Union: 99.35%) │
            └───────────────────┬────────────────────┘
                                │
                                ▼
            ┌────────────────────────────────────────┐
            │ Phase 6: Feature Engineering (25 Feats)│
            │  • Exact string & fuzzy metrics        │
            │  • High-throughput Sparse TF-IDF (23M/s)│
            │  • Address/Postal semantic alignment   │
            │  • Group context rankings & score gaps │
            └───────────────────┬────────────────────┘
                                │
                                ▼
            ┌────────────────────────────────────────┐
            │ Phase 7: Gradient Boosted Matcher      │
            │  • LightGBM pairwise binary ranker     │
            │  • Tiered negative sampling (15:1)     │
            │  • Macro F0.5 global threshold tuning  │
            └───────────────────┬────────────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │ Submission Output TSVs │
                    │ • matching_results.tsv │
                    │ • candidate_pairs.tsv  │
                    └────────────────────────┘
```

---

## 🚀 Key Innovations & Highlights

1. **Cross-Script Transliteration & Phonetic Reduction:**
   - Seamlessly resolves Latin representations against Indic scripts (Devanagari, Tamil, Gujarati, Bengali, etc.).
   - Employs an additive phonetic reduction layer that increased ground-truth cross-script string similarity from **0.59 to 0.75**.

2. **High-Recall Multi-Strategy Blocking:**
   - Combines token-level indexing, phonetic hash bucketing, regional postal-grouping, and address-aware pre-scoring.
   - **Post-Capped Blocking Recall (Model Candidates):** **98.28%** ($33,992 / 34,588$ confirmed validation matches preserved after top-K capping).
   - **Pre-Capping Theoretical Union Recall:** **99.35%** ($34,364 / 34,588$ raw candidates retrieved prior to capping).

3. **Ultra-Fast Sparse TF-IDF Pairwise Engine:**
   - Vectorized sparse matrix cosine calculations executing at **>23 Million pairs/sec**.

4. **Tiered Negative Sampling & Macro $F_{0.5}$ Thresholding:**
   - Preserves all true matches and hard exact-tier negatives while sub-sampling fuzzy negatives at 15:1.
   - Calibrates decision thresholds specifically for the Macro $F_{0.5}$ metric to maximize precision while preserving high recall.

---

## 📂 Repository Structure

```
BizEntity-Resolution-ML/
├── data/
│   └── splits/                  # Official 80/20 train/val entity split indices
├── notebooks/
│   ├── 01_eda.ipynb             # In-depth Exploratory Data Analysis
│   └── eda_assets/              # Visualizations & distribution charts
├── scripts/
│   ├── run_eda_analysis.py      # Automated EDA generation script
│   ├── evaluate_blocking_recall.py
│   ├── experiment_phonetic_simplify.py
│   ├── generate_official_split.py
│   ├── run_feature_engineering_sample.py
│   ├── run_phase7_matcher_sample.py
│   └── validate_phonetic_module.py
├── src/
│   ├── normalize/               # Name, address, country, phonetic normalization
│   ├── block/                   # Token, phonetic, postal blocking & candidate generation
│   ├── features/                # 25-feature pairwise extraction & TF-IDF similarity
│   ├── model/                   # LightGBM training, prediction, & threshold tuning
│   └── pipeline/                # Entity-level validation & Macro F0.5 scoring
├── tests/                       # Complete Pytest test suite (64 unit tests)
├── Documentation_template.md    # Official Challenge Methodology Document
├── requirements.txt             # Pinned project dependencies
└── .gitignore                   # Clean ignore rules for datasets & models
```

---

## 🛠️ Installation & Setup

```bash
# Clone repository
git clone https://github.com/Smitakalluri15/BizEntity-Resolution-ML.git
cd BizEntity-Resolution-ML

# Create and activate virtual environment (optional)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🧪 Running Tests

The test suite covers normalization, phonetic reduction, blocking, feature extraction, LightGBM training, and Macro $F_{0.5}$ scoring:

```bash
PYTHONPATH=. pytest tests/ -v
```
*(All 64 unit tests pass in < 4 seconds).*

---

## 📄 License & Compliance

- Built strictly in compliance with the Amazon ML Challenge 2026 guidelines.
- **Zero external APIs or internet lookups** used in normalization or matching.
- Uses open-source MIT/Apache libraries (LightGBM, RapidFuzz, Scikit-Learn).
