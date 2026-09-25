# Phase 6: Feature Engineering Module

The Feature Engineering module converts raw candidate pairs `(source1_entity_id, candidate_entity_id)` and their normalized entity records into rich, numeric pairwise feature vectors for downstream classification and ranking in Phase 7.

---

## 1. Feature Architecture & Catalog

The pipeline produces **25 numeric feature signals** and **4 identifier/metadata columns** (29 columns total) across 5 distinct categories:

### A. Name Similarity Signals (`src/features/name_features.py`)
| Feature | Type | Description & Rationale |
| :--- | :---: | :--- |
| `name_exact_raw` | `int32` [0, 1] | Exact identity on raw, unnormalized business names. High precision signal. |
| `name_exact_norm` | `int32` [0, 1] | Exact identity on normalized business names (after legal suffix removal, transliteration, and lowercasing). |
| `name_levenshtein_ratio` | `float32` [0, 1] | Normalized Levenshtein ratio computed via `rapidfuzz` C-extension ($>2.7\text{M pairs/sec}$). |
| `name_jaccard_tokens` | `float32` [0, 1] | Jaccard similarity of normalized name token sets: $\frac{\|A \cap B\|}{\|A \cup B\|}$. Robust to word-order permutations. |
| `name_token_overlap_count` | `int32` $\ge 0$ | Integer count of shared tokens between Source-1 and candidate names. |
| `name_char_trigram_jaccard` | `float32` [0, 1] | Character trigram Jaccard similarity on normalized name strings. Captures subtle typos and character-level edits. |
| `name_phonetic_similarity` | `float32` [0, 1] | Character n-gram similarity on phonetically-reduced names from Phase 2.5. Closes the transliteration gap for Indic scripts. |
| `name_script_match` | `int32` [0, 1] | `1` if Source-1 and candidate have identical detected script; `0` if candidate was cross-script / transliterated. |

### B. Address & Postal Signals (`src/features/address_features.py`)
| Feature | Type | Description & Rationale |
| :--- | :---: | :--- |
| `address_exact_norm` | `int32` [0, 1] | Exact identity on normalized addresses (after abbreviation expansion and landmark stripping). |
| `address_levenshtein_ratio` | `float32` [0, 1] | Normalized Levenshtein ratio on normalized address strings. |
| `address_jaccard_tokens` | `float32` [0, 1] | Jaccard similarity on address token sets. |
| `address_char_trigram_jaccard` | `float32` [0, 1] | Character trigram Jaccard similarity on normalized addresses. |
| `postal_available` | `int32` [0, 1] | `1` if **both** Source-1 and candidate have non-null, valid postal codes; `0` otherwise. |
| `postal_exact_match` | `int32` [0, 1] | `1` if `postal_available == 1` and postal codes match exactly; `0` if mismatched or unavailable. |
| `landmark_flag_either` | `int32` [0, 1] | `1` if either record is an approximate landmark address (e.g. "near hospital", "opp bus stand"). Indicates address comparison noise. |

### C. Corpus-Wide Sparse TF-IDF Cosine Signals (`src/features/tfidf_features.py`)
| Feature | Type | Description & Rationale |
| :--- | :---: | :--- |
| `name_tfidf_word_cosine` | `float32` [0, 1] | Word 1-2 gram TF-IDF cosine similarity on business names across corpus vocabulary. |
| `name_tfidf_char_cosine` | `float32` [0, 1] | Character 2-4 gram TF-IDF cosine similarity on business names. Captures sub-word morphological similarity. |
| `address_tfidf_word_cosine` | `float32` [0, 1] | Word 1-2 gram TF-IDF cosine similarity on business addresses across corpus vocabulary. |

### D. Cross-Interaction & Group-Context Signals (`src/features/cross_features.py`)
| Feature | Type | Description & Rationale |
| :--- | :---: | :--- |
| `country_match` | `int32` [0, 1] | `1` if countries match or either is missing; sanity check verifying Phase 5 country partition. |
| `name_address_interaction` | `float32` [0, 1] | Product of `name_jaccard_tokens` $\times$ `address_jaccard_tokens`. Differentiates pairs strong in both vs strong in only one. |
| `candidate_rank_in_group` | `int32` $\ge 1$ | 1-based rank of the candidate within its Source-1 entity candidate group (sorted by similarity score descending). |
| `candidate_set_size` | `int32` $\ge 1$ | Total count of candidates surfaced for this Source-1 entity. |
| `top1_gap` | `float32` $\ge 0$ | Score difference between the rank 1 and rank 2 candidate in the S1 group. Large gap = high distinction confidence. |
| `candidate_source_id` | `int32` [0, 1] | `0` for Source-2 candidates, `1` for Source-3 candidates. |

### E. Tiered Retention & Provenance Metadata (`src/features/cross_features.py`)
| Field | Type | Description |
| :--- | :---: | :--- |
| `candidate_source_strategy` | `object` (str) | Blocking strategy provenance: `'token'`, `'postal'`, `'phonetic'`, `'sorted_neighborhood'`, or multi-strategy combinations. |
| `is_exact_tier` | `int32` [0, 1] | `1` if surfaced by exact token or exact postal match; `0` if surfaced only by fuzzy phonetic / sorted neighborhood. |

---

## 2. Sparse TF-IDF Optimization & Performance

### The High-Scale Challenge
Computing pairwise cosine similarity across $50\text{--}150\text{M}$ candidate pairs naively with dense matrices or per-pair sklearn calls is computationally infeasible.

### The Sparse Matrix Vectorization Solution
1. **L2 Unit Norm Transformation:** `TfidfVectorizer(norm='l2')` transforms texts into unit-length sparse rows ($||\mathbf{u}||_2 = 1$).
2. **Elementwise Sparse Multiplication:**
   $$\text{CosineSimilarity}(\mathbf{u}_i, \mathbf{v}_i) = \mathbf{u}_i \cdot \mathbf{v}_i = \sum_{j} (\mathbf{u}_i \odot \mathbf{v}_i)_j$$
3. **Execution Throughput:**
   - Sparse pairwise dot product evaluates at **$>23\text{ Million pairs/second}$**.
   - Total feature pipeline executes at **$\approx 6,534\text{ pairs/second}$** on a single thread ($\approx 45\text{ seconds}$ for $300,000$ pairs).

---

## 3. Label Assembly & Class Balance Analysis

- **Ground Truth Evaluation:** Labels are assembled using `src/features/build_features.assemble_labels()` with strictly held-out validation IDs (`val_ids.txt` remains unobserved).
- **Severe Class Imbalance:**
  - Candidate generation deliberately generates $50\text{--}60$ candidates per S1 entity to achieve $\ge 99\%$ blocking recall.
  - As a result, positive pairs (`is_true_match == 1`) represent **$\approx 0.04\%\text{--}0.08\%$** of all candidate pairs.
  - Imbalance Ratio: **$\approx 1,500\text{ : }1$ to $2,300\text{ : }1$** (Negative : Positive).

---

## 4. Downstream Guidance for Phase 7 (Classification & Ranking)

1. **Macro $F_{0.5}$ Alignment:** The competition metric weights precision $4\times$ more than recall ($\beta=0.5$). The model must learn conservative decision thresholds.
2. **Tiered Negative Downsampling:** During training, keep 100% of positive pairs and exact-tier negatives (`is_exact_tier == 1`), while subsampling fuzzy negatives (`is_exact_tier == 0`) to achieve a balanced training ratio of $\approx 10\text{ : }1$ or $20\text{ : }1$.
3. **Group Ranking Objective:** Utilize LightGBM/XGBoost `lambdarank` or binary classification with `group` parameter mapped to `source1_entity_id`.
