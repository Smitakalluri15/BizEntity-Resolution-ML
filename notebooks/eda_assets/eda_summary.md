# Business Entity Resolution — Exploratory Data Analysis (EDA) Summary
**Amazon ML Challenge 2026 — Phase 1 Consolidated Findings**

---

## 1. Executive Summary & Dataset Dimensions

The challenge objective is to match noisy external business records from **Source 2 (S2)** and **Source 3 (S3)** against a clean, deduplicated reference list in **Source 1 (S1)**. The evaluation metric is **Macro $F_{0.5}$**, which weights precision twice as heavily as recall (penalizing false matches ~2x more than missed matches).

| Dataset File | Role | Total Rows | Total Columns | Missing Values / Nulls |
| :--- | :--- | :--- | :--- | :--- |
| `train_source1.tsv` | Canonical Reference | **2,206,821** | 4 | 0 nulls across all fields |
| `train_source2.tsv` | External Source 2 | **5,034,616** | 4 | `business_name`: 2 nulls<br>`business_address`: 168,967 nulls (3.36%) |
| `train_source3.tsv` | External Source 3 | **5,285,603** | 4 | `business_name`: 13 nulls<br>`business_address`: 175,916 nulls (3.33%) |
| `train_ground_truth.tsv` | True Match Labels | **2,206,821** | 2 | `matched_entity_ids`: 123,247 empty (singletons) |

---

## 2. Ground Truth Structure & Match-Count Distribution

Every Source 1 entity in `train_source1.tsv` is represented exactly once in `train_ground_truth.tsv`.

### Key Metrics:
- **Total S1 Reference Entities:** 2,206,821 (100.0%)
- **Singletons (0 Matches):** **123,247 (5.58%)** — These S1 records have zero matching records in S2 or S3.
- **Exactly 1 Match:** **119,157 (5.40%)**
- **Two or More Matches (Multi-Match):** **1,964,417 (89.02%)**
- **Total Matched Entity Links:** **7,638,365**
  - Matched to Source 2: **3,693,619 (48.36%)**
  - Matched to Source 3: **3,944,746 (51.64%)**
  - Invalid / Other Prefixes: **0 (0.00%)**

### Full Match-Count Histogram:
| Matched Entity Count | S1 Entity Count | Percentage | Cumulative % |
| :---: | :---: | :---: | :---: |
| **0** (Singletons) | 123,247 | 5.58% | 5.58% |
| **1** | 119,157 | 5.40% | 10.99% |
| **2** | 375,212 | 17.00% | 27.99% |
| **3** | 530,841 | 24.05% | 52.04% |
| **4** | 484,115 | 21.94% | 73.98% |
| **5** | 321,957 | 14.59% | 88.57% |
| **6** | 164,868 | 7.47% | 96.04% |
| **7** | 63,968 | 2.90% | 98.94% |
| **8** | 18,680 | 0.85% | 99.78% |
| **9** | 4,205 | 0.19% | 99.98% |
| **10** | 534 | 0.02% | 100.00% |
| **11** | 37 | < 0.01% | 100.00% |

> **Strategic Takeaway:** 89% of S1 entities have multiple matches across S2 and S3, with 3 to 4 matches being the most common mode. However, because Macro $F_{0.5}$ penalizes false positives heavily, singletons (5.58%) and low-confidence candidate links must be conservatively filtered.

---

## 3. Data Integrity & Noise Rates

### Data Quality Sanity Checks:
- **Duplicate IDs:** **0 duplicates** in S1, S2, S3, or GT. Every `entity_id` is unique.
- **Prefix Consistency:** 100% compliant (`S1-`, `S2-`, `S3-`).
- **Orphaned References:** **0 orphaned references**. All 7,638,365 matched entity IDs in the ground truth exist in `train_source2.tsv` or `train_source3.tsv`.
- **S1 Consistency:** 100% 1-to-1 match between S1 table and Ground Truth table (no missing or extra S1 IDs).

### Unreferenced "Noise" Records (Distractors):
In Source 2 and Source 3, a significant portion of records are distractors that do **not** match any Source 1 business entity:
- **Source 2 Noise Rate:** **26.64%** (1,340,997 unreferenced records out of 5,034,616)
- **Source 3 Noise Rate:** **25.37%** (1,340,857 unreferenced records out of 5,285,603)
- **Combined External Noise Records:** **2,681,854 distractor records**.

---

## 4. Country Distribution & Critical Generalization Warning

### Training Country Breakdown:
| Country | Source 1 Count (%) | Source 2 Count (%) | Source 3 Count (%) |
| :--- | :--- | :--- | :--- |
| **US** | 1,323,633 (59.98%) | 3,016,817 (59.92%) | 3,170,056 (59.98%) |
| **India** | 883,188 (40.02%) | 2,017,799 (40.08%) | 2,115,547 (40.02%) |

> [!WARNING]
> **CRITICAL ARCHITECTURAL CONSTRAINT: UNSEEN COUNTRIES IN TEST SET**
> The training data strictly contains `US` and `India`. However, official competition specifications state that the **test set contains at least one additional country (`France`) never seen in training**.
> 
> **Downstream Implementation Rules:**
> 1. **No Hardcoded Allow-Lists:** Never hardcode `['US', 'India']` in any filtering, blocking, or one-hot encoding logic.
> 2. **Country-Agnostic Normalization:** Address and name parsing must gracefully handle international patterns (e.g., French postal codes, European legal suffixes like *SAS*, *SARL*).
> 3. **Exact Country Blocking:** Country equality (i.e. `country_s1 == country_s2`) remains a high-precision candidate blocking constraint regardless of the specific country string.

---

## 5. Top Impactful Noise Patterns & Normalization Requirements

From manual inspection of sampled pairs across all match buckets, the following 6 noise categories represent the largest sources of divergence:

1. **Indic & Multilingual Script Divergence (High Impact for India):**
   - *Pattern:* S1 is transliterated in Latin script, while matching S2/S3 records are written in Indic scripts (Devanagari, Gujarati, Tamil, Kannada).
   - *Example:* S1 `"Great Engineering Limited"` matches S3 `"ग्रेट इंजीनियरिंग लिमिटेड"`.
   - *Downstream Rule:* A phonetic/transliteration module or multilingual text embedding model is required to bridge the script gap.

2. **Web URLs, Social Handles, and Domain Names in Entity Names:**
   - *Pattern:* S2/S3 names often contain domain names, website URLs (`.com`, `.org`), or `@handles`.
   - *Example:* S1 `"Solora Wave LLC"` matches S3 `"wavesolora.com"`; S1 `"Quality Asset Solutions, LLC"` matches S3 `"@qualityasset"`.
   - *Downstream Rule:* Normalize domains by stripping `www.`, `.com`, `@`, and splitting camelCase or concatenated words.

3. **OCR / Typographic Digit-for-Letter Substitutions & Accents:**
   - *Pattern:* Visual/OCR typos (`5` $\to$ `S`, `6` $\to$ `G`, `0` $\to$ `O`, `l` $\to$ `I`) and accented characters (`Á`, `Í`).
   - *Example:* S1 `"Solutions Al Spaces Center"` matches S3 `"5olutions Al Spaces Center"`; S1 `"RK Great Advisors LLC"` matches S2 `"RK 6reat Advisors LLC"`.
   - *Downstream Rule:* Standardize character sets with ASCII folding (`unicodedata.normalize`) and token-level fuzzy / character n-gram similarity (Jaro-Winkler, Levenshtein, bi-gram Jaccard).

4. **Missing Addresses & Truncation in External Sources:**
   - *Pattern:* Over **344,000 external records (~3.35%)** have completely null/NaN address fields, while others omit street numbers or cities.
   - *Example:* S1 `"1739 Labrador Drive, Costa Mesa, CA"` matches S3 with address `nan`.
   - *Downstream Rule:* Name matching must be robust enough to operate independently when address signals are completely absent.

5. **Legal Suffix Inconsistency & Word-Order Transposition:**
   - *Pattern:* Legal suffixes are omitted, expanded, abbreviated (`Ltd`, `Pvt Ltd`, `LLC`, `Corp`), or moved to the front.
   - *Example:* S1 `"Peridos Investment LLC"` matches S2 `"LLC Peridos Ínvestment"`.
   - *Downstream Rule:* Suffix normalization and set-based token overlap (e.g. Token Sort / Token Set Ratio) are necessary.

6. **Landmark References & Component Reordering in Addresses:**
   - *Pattern:* Addresses contain relative landmarks (`Near SBI ATM`, `Behind Hinduja College`, `Opposite Post Office`) and unordered components.
   - *Example:* S1 `"Pl 42 New Mangalwar Peth, Nr Ladka T Pump, Pune, Maharashtra"` matches S3 `"Block F-496 Pl 42 New Mangalwar Peth, Nr Ladka T Pump, Pune, महाराष्ट्र"`.
   - *Downstream Rule:* Robust n-gram matching and landmark stopword handling.

---

## 6. Token & Character Length Statistics

| Dataset | Field | Mean Char Len | Median Char Len | Min / Max Char Len | Mean Tokens | Median Tokens | Min / Max Tokens |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Source 1** | `business_name` | 24.03 | 24.0 | 3 / 105 | 3.55 | 4.0 | 1 / 16 |
| **Source 2** | `business_name` | 25.10 | 25.0 | 0 / 104 | 3.61 | 4.0 | 0 / 16 |
| **Source 3** | `business_name` | 25.20 | 25.0 | 0 / 123 | 3.63 | 4.0 | 0 / 18 |
| **Source 1** | `business_address` | 52.07 | 41.0 | 11 / 256 | 8.03 | 7.0 | 2 / 43 |
| **Source 2** | `business_address` | 46.23 | 37.0 | 0 / 249 | 7.31 | 6.0 | 0 / 46 |
| **Source 3** | `business_address` | 46.71 | 42.0 | 0 / 240 | 7.17 | 6.0 | 0 / 43 |

### Blocking Strategy Implications:
- Business names are short (median 4 tokens, ~25 chars). Standard sorted-neighborhood blocking windows on name prefixes must account for token transposition (e.g., prefix blocking on the first token alone will miss `LLC Peridos Investment`).
- Multi-token combinations (e.g., Sorted First 2 Tokens + Country, or Trigram MinHash LSH) will be necessary.

---

## 7. Anomalies & Actionable Decisions for Next Phases

1. **Null Address Handling:** 3.3% of S2/S3 records have missing addresses. Pipelines cannot rely solely on address matching and must have a fallback name-only similarity threshold.
2. **True Noise vs Unmatched Entities:** 25-27% of S2 and S3 records have no match in S1. Candidate generation must reject these aggressively to protect precision.
3. **Indic Transliteration Strategy:** Because ~40% of records are from India and many S2/S3 records use Indic scripts, implementing an Indic-to-Latin transliteration pre-processing step (or multilingual embedding encoder) is essential in Phase 2/3.
4. **All Outputs & Assets:** Visualizations are saved in `notebooks/eda_assets/`, and the full notebook is executable at `notebooks/01_eda.ipynb`.
