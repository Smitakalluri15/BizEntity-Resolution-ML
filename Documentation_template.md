# ML Challenge 2026: Business Entity Resolution Solution Documentation

**Team Name:** BizEntity Masters  
**Submission Date:** 2026-09-25  

---

## 1. Executive Summary
We designed a high-throughput, cross-lingual Business Entity Resolution system capable of matching reference entities (Source 1) against noisy, multi-source records (Source 2 & Source 3) across 12.5M+ rows. Our solution integrates: (1) an additive phonetic reduction and Indic transliteration layer, (2) a high-recall multi-strategy inverted-index blocking engine achieving **98.28% post-capped model candidate recall** (with a **99.35% pre-capping theoretical union ceiling**), and (3) a LightGBM pairwise binary matcher trained on 25 pairwise & group-context features calibrated directly for the **Macro $F_{0.5}$** objective.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis across 12.52 million records revealed three primary challenges:
1. **Cross-Script & Transliteration Discrepancies:** Businesses in India and multi-lingual regions appear interchangeably in native Indic scripts (Devanagari, Tamil, Gujarati, Bengali) and Latin Romanizations with frequent phonological variations.
2. **Legal Entity Transpositions & OCR Noise:** Frequent swapping of corporate identifiers (`Pvt Ltd` $\leftrightarrow$ `Private Limited`, `LLC`, `GmbH`, `S.A.S`, `SCI`) and digit-letter OCR confusions (`0` $\leftrightarrow$ `O`, `1` $\leftrightarrow$ `I`).
3. **Severe Class Imbalance & Evaluation Asymmetry:** With 12.5M records, negative pair combinations exceed $10^{13}$. The competition metric, Macro $F_{0.5}$, weights precision **$4\times$ higher than recall**, requiring aggressive suppression of false-positive merges while preserving exact singletons.

### 2.2 Solution Strategy
- **Approach Type:** Hybrid Multi-Strategy Inverted-Index Blocking + High-Speed Sparse TF-IDF + Gradient Boosted Decision Tree (LightGBM) Pairwise Ranker.
- **Core Innovations:**
  - **Additive Phonetic Reduction Layer:** Standardizes Indic sound classes (aspirated vs. unaspirated stops, sibilants, vowel length invariance) post-transliteration, boosting cross-script similarity from $0.59 \to 0.75$.
  - **Vectorized Sparse TF-IDF Cosine Engine:** Custom matrix algebra computing character/word n-gram similarities at $>23\text{ Million pairs/sec}$.
  - **Tiered Negative Sampling & Asymmetric Calibration:** Preserves hard exact-tier negatives while sub-sampling loose candidates at 15:1, tuned specifically for global Macro $F_{0.5}$ maximization.

---

## 3. Candidate Generation (Blocking)

To reduce the $12.5\text{M} \times 12.5\text{M}$ comparison space without dropping true matches:
- **Blocking Keys Used:**
  1. *Token-based Inverted Index:* Standardized name tokens with dynamic stopword frequency thresholding ($\le 10,000$).
  2. *Phonetic Soundex & 3-Gram Hash Index:* Inverted buckets for phonetically reduced name representations.
  3. *Postal Code & Geographic Blocks:* Exact 6-digit PIN / 5-digit ZIP prefixes combined with normalized open-set country partitions.
  4. *Sorted Neighborhood Partitions:* Alphabetical sliding windows (window size $W=10$).
- **Candidate Reduction & Pre-Scoring:** Symmetrically pre-scores candidates via RapidFuzz token set ratio in reduced phonetic space, address token overlap, postal match, and cross-script prioritization, truncating to top-K ($K=450$).
- **Empirical Validation (Held-out Validation Split):**
  - **Post-Capped Blocking Recall (Handed to Model):** **98.28%** ($33,992 / 34,588$ confirmed ground-truth matches retrieved).
  - **Pre-Capping Theoretical Union Recall:** **99.35%** ($34,364 / 34,588$ matches caught in raw candidate union before truncation).

---

## 4. Matching Model

### 4.1 Feature Set (25 Features)
| Feature Group | Features | Description |
|---|---|---|
| **Name Similarities** | `name_exact_match`, `name_norm_exact_match`, `name_token_jaccard`, `name_levenshtein_ratio`, `name_token_sort_ratio`, `name_phonetic_similarity`, `name_char_trigram_jaccard`, `name_length_ratio`, `name_prefix_match_3` | RapidFuzz C-accelerated Levenshtein, token set ratios, phonetic string distance, character n-grams. |
| **Address & Geo** | `address_token_jaccard`, `address_levenshtein_ratio`, `postal_exact_match`, `postal_prefix_match`, `country_match`, `is_landmark_address` | Semantic token overlap, edit distance, postal prefix alignment, open-set country equality. |
| **TF-IDF Similarities** | `name_tfidf_word_cosine`, `name_tfidf_char_cosine`, `addr_tfidf_word_cosine`, `addr_tfidf_char_cosine` | Vectorized sparse matrix cosine similarities on word and character 3-5 n-grams. |
| **Group Context & Cross Terms** | `cand_rank_in_group`, `cand_score_gap_to_top`, `cand_score_gap_to_next`, `name_x_address_sim`, `is_exact_tier` | Relative confidence within entity candidate pool, margin to rival candidate, interaction terms. |

### 4.2 Model & Decision Policy
- **Model Architecture:** LightGBM Binary Classifier (200 trees, max depth 6, learning rate 0.05, `binary_logloss`).
- **Threshold Selection:** Exhaustive grid search over $[0.05, 0.95]$ maximizing Macro $F_{0.5}$ directly on held-out validation entities.
- **Winning Policy:** Multi-match emission with optimal threshold **$\theta^* = 0.60$**. If no candidate clears $0.60$, the system emits an empty string `""` (singleton).

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score (Validation):** **$0.0656$** (Sample-scale validation with multi-match threshold $\theta^*=0.60$, representing a **$+17.4\%$ improvement** over the trivial singleton floor of $0.0558$).
- **Predicted Probability Separation:**
  - True Matches: **Median probability $1.0000$** ($84.6\%$ scored with $\ge 0.99$ confidence).
  - Negative Candidates: **Median probability $0.0000$** ($99.86\%$ filtered below $0.50$).
- **Common False Positives:** Highly generic business chains sharing a common brand name but distinct local branch postal codes when address fields are omitted.
- **Common False Negatives:** Severely truncated Tamil business names where transliteration ambiguity and missing postal codes prevented token overlap.

---

## 6. Conclusion
Our solution demonstrates that combining domain-specific phonetic normalization, high-recall inverted index blocking, and an asymmetrically tuned gradient-boosted ranker provides a scalable, noise-resilient entity resolution pipeline. The system operates strictly within memory limits without external lookups, fully complying with Amazon ML Challenge 2026 specifications.

---

## Appendix

### A. Code Artefacts & Structure
The submission package ships under `code/business_entity_resolution/`:
```
code/business_entity_resolution/
├── src/
│   ├── normalize/        # Name, address, country, transliteration, phonetic modules
│   ├── block/            # Token, phonetic, postal blocking & candidate generation
│   ├── features/         # 25-feature pairwise extraction & TF-IDF similarity
│   ├── model/            # LightGBM training, inference, & threshold tuning
│   └── pipeline/         # Validation split & Macro F0.5 scoring harness
├── scripts/
│   ├── run_submission_pipeline.py           # Full test inference pipeline
│   └── validate_and_package_submission.py   # Integrity check and zip builder
├── requirements.txt
└── README.md
```
- **Execution Entry Point:** `python scripts/run_submission_pipeline.py --test_dir dataset/test --output_dir output`
