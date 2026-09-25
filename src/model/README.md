# Phase 7: Gradient Boosted Entity Matcher (Sample-Scale Dev/Validation)

The Gradient Boosted Matcher module trains and evaluates high-precision pairwise ranking and binary classification models for Business Entity Resolution under the competition's **Macro $F_{0.5}$** metric.

---

## 1. Pipeline Overview & Architecture

The Phase 7 pipeline consists of 4 core modules:
1. **Tiered Negative Sampling ([src/model/prepare_training_data.py](file:///Users/smitakalluri/amazon_project/src/model/prepare_training_data.py)):**
   - Strictly isolates training S1 entities (`train_ids.txt`) from validation entities (`val_ids.txt`).
   - Retains 100% of true positive pairs (`is_true_match == 1`).
   - Retains 100% of exact-tier negatives (`is_true_match == 0` and `is_exact_tier == 1`) to train the model on confusable near-misses.
   - Subsamples fuzzy-tier negatives down to a controlled target ratio.
2. **GBDT Model Training ([src/model/train.py](file:///Users/smitakalluri/amazon_project/src/model/train.py)):**
   - Trains LightGBM Binary Classifier on 25 pairwise numeric features.
   - Saves trained model artifact to `models/phase7_matcher_sample.txt`.
3. **Entity Prediction Assembly ([src/model/predict.py](file:///Users/smitakalluri/amazon_project/src/model/predict.py)):**
   - Predicts pairwise match probabilities.
   - Assembles entity predictions in the official `matching_results.tsv` format supporting both `mode="multi"` and `mode="top1"`.
4. **Validation Threshold Tuning ([src/model/threshold_tuning.py](file:///Users/smitakalluri/amazon_project/src/model/threshold_tuning.py)):**
   - Sweeps decision thresholds $[0.05, 0.95]$ against held-out validation ground truth using Phase 3's exact `compute_macro_f_beta()` scorer.

---

## 2. Sample-Scale Training & Class Balance

- **Input Dataset:** `data/features/sample_candidate_features_labeled.parquet` ($299,352$ candidate pairs across $5,000$ S1 entities).
- **Split Distribution in Sample:**
  - Training S1 Entities: $3,992$ ($193,047$ candidate pairs)
  - Validation S1 Entities: $1,008$ ($60,357$ candidate pairs)
- **Class Balance & Sampling:**
  - True Positive Pairs ($N_{\text{pos}}$): $104$
  - Exact-tier Negatives Retained: $192,943$ ($100\%$ retained)
  - Fuzzy-tier Negatives Subsampled: $0$
  - Effective Training Balance: $1,855.22\text{ : }1$ (Neg : Pos)

---

## 3. LightGBM Hyperparameters (Provisional Sample Configuration)

| Parameter | Value | Rationale |
| :--- | :---: | :--- |
| `objective` | `'binary'` | Pairwise binary classification (match vs non-match). |
| `metric` | `'binary_logloss'` | Probabilistic calibration for downstream thresholding. |
| `learning_rate` | `0.05` | Stable gradient stepping. |
| `num_leaves` | `31` | Controls tree complexity. |
| `n_estimators` | `300` | Tree ensemble capacity. |
| `subsample` | `0.8` | Row subsampling for regularization. |
| `colsample_bytree` | `0.8` | Feature subsampling. |

*Note: These hyperparameters are provisional and designed for sample-scale development. Full hyperparameter optimization will be performed on full-scale data.*

---

## 4. Feature Importance Rankings

Trained LightGBM feature splits on the sample dataset:

| Rank | Feature Name | Split Count | Importance (%) | Signal Category |
| :---: | :--- | :---: | :---: | :--- |
| **01** | `name_address_interaction` | $956$ | **$10.63\%$** | Cross-field interaction |
| **02** | `top1_gap` | $948$ | **$10.54\%$** | Group context / margin |
| **03** | `address_levenshtein_ratio` | $892$ | **$9.92\%$** | Address string similarity |
| **04** | `address_jaccard_tokens` | $789$ | **$8.77\%$** | Address token overlap |
| **05** | `name_tfidf_char_cosine` | $699$ | **$7.77\%$** | Sub-word char TF-IDF cosine |
| **06** | `name_tfidf_word_cosine` | $672$ | **$7.47\%$** | Word TF-IDF cosine |
| **07** | `candidate_rank_in_group` | $618$ | **$6.87\%$** | Group context rank |
| **08** | `address_char_trigram_jaccard` | $551$ | **$6.13\%$** | Address trigram similarity |
| **09** | `name_levenshtein_ratio` | $543$ | **$6.04\%$** | Name string distance |
| **10** | `name_phonetic_similarity` | $537$ | **$5.97\%$** | Cross-script phonetic similarity |
| **11** | `name_char_trigram_jaccard` | $518$ | **$5.76\%$** | Name char trigram Jaccard |
| **12** | `address_tfidf_word_cosine` | $403$ | **$4.48\%$** | Address TF-IDF cosine |
| **13** | `name_jaccard_tokens` | $303$ | **$3.37\%$** | Name token Jaccard |
| **14** | `name_token_overlap_count` | $190$ | **$2.11\%$** | Shared token count |
| **15** | `candidate_source_id` | $139$ | **$1.55\%$** | Source origin (S2 vs S3) |
| **16** | `landmark_flag_either` | $80$ | **$0.89\%$** | Landmark noise indicator |
| **17** | `name_script_match` | $55$ | **$0.61\%$** | Cross-script flag |
| **18** | `name_exact_norm` | $51$ | **$0.57\%$** | Exact normalized name match |
| **19** | `name_exact_raw` | $19$ | **$0.21\%$** | Exact raw name match |
| **20** | `candidate_set_size` | $18$ | **$0.20\%$** | S1 candidate set volume |
| **21** | `postal_available` | $12$ | **$0.13\%$** | Postal code availability |

---

## 5. Validation Threshold Sweep Results

Evaluated on $1,008$ validation S1 entities ($60,357$ candidate pairs):

| Threshold | Mode | Macro Precision | Macro Recall | Macro $F_{0.5}$ |
| :---: | :---: | :---: | :---: | :---: |
| $0.10$ | `multi` | $0.0734$ | $0.0585$ | $0.0654$ |
| $0.10$ | `top1` | $0.0734$ | $0.0582$ | $0.0650$ |
| $0.30$ | `multi` | $0.0734$ | $0.0585$ | $0.0654$ |
| $0.30$ | `top1` | $0.0734$ | $0.0582$ | $0.0650$ |
| $0.50$ | `multi` | $0.0734$ | $0.0585$ | $0.0654$ |
| $0.50$ | `top1` | $0.0744$ | $0.0584$ | $0.0654$ |
| **0.60** | **`multi`** | **$0.0739$** | **$0.0585$** | **$0.0656$** |
| $0.60$ | `top1` | $0.0734$ | $0.0582$ | $0.0650$ |
| $0.80$ | `multi` | $0.0739$ | $0.0585$ | $0.0656$ |
| $0.80$ | `top1` | $0.0744$ | $0.0584$ | $0.0654$ |
| $0.90$ | `multi` | $0.0739$ | $0.0585$ | $0.0656$ |
| $0.90$ | `top1` | $0.0744$ | $0.0584$ | $0.0654$ |

### Configuration Comparison
- **Best Multi-Match Configuration:** $\text{Threshold} = 0.60 \implies \text{Macro } F_{0.5} = \mathbf{0.065551}$
- **Best Top-1 Configuration:** $\text{Threshold} = 0.50 \implies \text{Macro } F_{0.5} = 0.065384$
- **Trivial Baseline (Predict Empty):** $\text{Macro } F_{0.5} = 0.055847$
- **Net Relative Improvement:** **$+17.4\%$** over baseline on sample validation slice.

---

## 6. Important Disclaimer & Mandatory Next Steps

> [!WARNING]
> **This phase's results are directionally useful only.** The sample-scale numbers are computed on a small, non-representative slice of the validation set ($1,008$ out of $441,363$ entities).
> 
> **Next steps before final submission:**
> 1. Run the full-scale Phase 6 feature build across all $2.2\text{M}$ Source-1 records ($\approx 110\text{--}130\text{M}$ candidate pairs).
> 2. Re-run this Phase 7 training pipeline end-to-end against the full-scale feature dataset.
> 3. Re-tune the decision threshold on the full $441,363$-entity validation set before treating any model threshold as final.
