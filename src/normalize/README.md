# Business Entity Resolution — Normalization Module (`src/normalize/`)
**Amazon ML Challenge 2026 — Phase 2 Documentation**

---

## 1. Overview & Architecture

The normalization module transforms noisy, heterogeneously formatted business records across **Source 1 (Reference)**, **Source 2**, and **Source 3 (External)** into clean, standardized representation vectors suitable for candidate generation, blocking, and similarity scoring.

### Core Architectural Principles:
1. **Preserve Raw Fields:** Original columns (`business_name`, `business_address`, `country`) are never overwritten. Normalized variants (`name_norm`, `address_norm`, `country_norm`, etc.) and auxiliary feature flags (`name_script`, `name_embedded_url`, `is_landmark_address`, `postal_code`) are added alongside.
2. **Cross-Script Romanization First:** Because ~40% of records are from India and external sources contain records written in Indic scripts (Devanagari, Gujarati, Tamil, etc.), script detection and transliteration occur **before** any alphanumeric normalization.
3. **No Country Allowlists (Open-Set Generalization):** Training data covers `US` and `India`, while test data introduces unseen countries (e.g. `France`). Country normalization is strictly string-based without hardcoded allowlists.
4. **Vectorized High-Throughput Processing:** The batch module (`apply_normalization`) uses single-pass unique string dictionary mapping, achieving **> 32,000 records/second** (~67s for 2.2M Source 1 records).

---

## 2. Transformation Rules & EDA Justifications

| Pipeline Step | Specific Transformation | EDA Finding & Justification (`eda_summary.md`) |
| :--- | :--- | :--- |
| **Script Detection & Transliteration** | Unicode code-point detection $\to$ Romanization via `indic-transliteration` (`ITRANS` scheme). | S1 is in Latin English, while S2/S3 contain matching businesses in Devanagari (`ग्रेट इंजीनियरिंग`), Gujarati (`માય પાવર`), and Tamil (`ஸ்ரீ பவர்`). String distance across different Unicode blocks is 0% without transliteration. |
| **URL & Handle Extraction** | Regex detection of URLs (`.com`, `www.*`, `.org`, `.in`) and `@handles` $\to$ stored in `name_embedded_url`, stripped from `name_norm`. | S2/S3 records frequently contain domain names (`wavesolora.com`), handles (`@qualityasset`), or website links (`www.maladexpo.com`) appended to or replacing names. |
| **Contextual OCR Repair** | Context-bounded digit-to-letter repair: `5olutions` $\to$ `solutions`, `6reat` $\to$ `great`. | S2/S3 exhibits OCR digit substitutions. Positive lookaround ensures legitimate alphanumeric tokens (`7-Eleven`, `3M`, `2598 Center St`) are protected. |
| **Diacritic Normalization** | Unicode `NFKD` decomposition + combining character stripping (`Á` $\to$ `A`, `Í` $\to$ `I`). | S2/S3 records contain stray accents (e.g. `BÁY COLOMBIER`, `LLC Peridos Ínvestment`, `Smart Tráilblazer`). |
| **Ampersand & Conjunction Normalization** | Convert `&` and `+` into `' and '`. | Conjunction variations are prevalent across sources (`Valencia, Knicely & Prado` vs `Valencia Knicely and Prado`). |
| **Legal Suffix Stripping (Position-Agnostic)** | Word-boundary removal of legal entity tokens across the entire string (start, middle, end, brackets). | S2/S3 records frequently drop, expand, or **transpose suffixes to the front** (`LLC Peridos Investment` vs `Peridos Investment LLC`, `Sanskruti Technologies Pvt (Ltd)`). |
| **Address Abbreviation Expansion** | Standard postal expansions (`st` $\to$ `street`, `rd` $\to$ `road`, `ave` $\to$ `avenue`, `nr` $\to$ `near`, `fl` $\to$ `floor`, etc.). | Sources alternate between full words and abbreviations (`2756 GREELEY ST` vs `2756 Greeley Street`). |
| **Landmark Address Flagging** | Keyword detector (`near`, `behind`, `nr`, `opp`, `ke pass`, `c/o`) $\to$ boolean `is_landmark_address`. | Indian records frequently use landmarks (`Nr Ladka T Pump`, `Behind Hinduja College`) which carry lower street-grid precision and need name-prioritized matching. |
| **Postal / PIN Code Extraction** | Strict regex extraction of US 5/9-digit ZIP and Indian 6-digit PIN codes $\to$ `postal_code`. | Enables exact high-precision blocking when PIN/ZIP codes are present. |
| **Null & Placeholder Scrubbing** | Handles `None`, `NaN`, `null`, `<NULL>`, `nan`, `N/A` gracefully $\to$ `""`. | ~3.35% of S2 and S3 records (over 344,000 rows) have completely missing addresses. |

---

## 3. Finalized Dictionaries & Regex Patterns

### Legal Suffixes (`LEGAL_SUFFIXES` in `src/normalize/name.py`)
```python
[
    # English / US / Commonwealth
    'incorporated', 'inc', 'corporation', 'corp',
    'private limited', 'pvt ltd', 'pvt limited', 'private ltd',
    'limited', 'ltd', 'private', 'pvt',
    'limited liability company', 'limited liability partnership',
    'pllc', 'llc', 'llp', 'lp', 'company', 'co',
    'public limited company', 'plc',
    # European / International (for unseen test countries like France)
    'gmbh', 'sarl', 'sas', 'sa', 'bv', 'nv', 'srl', 'spa',
    # Transliterated Indic Suffixes (from ITRANS romanization)
    'limiteDa', 'limiteda', 'limidhedh', 'limidhed',
    'prAiveTa', 'praiveta', 'bhiraivedh', 'bhiraived'
]
```

### Address Abbreviations (`ADDRESS_ABBREVIATIONS` in `src/normalize/address.py`)
```python
{
    'st': 'street', 'str': 'street',
    'rd': 'road',
    'ave': 'avenue', 'av': 'avenue',
    'apt': 'apartment',
    'blvd': 'boulevard',
    'dr': 'drive',
    'ln': 'lane',
    'ct': 'court',
    'fl': 'floor', 'flr': 'floor',
    'ste': 'suite',
    'bldg': 'building',
    'pkwy': 'parkway',
    'pl': 'place',
    'hwy': 'highway',
    'nr': 'near',
    'opp': 'opposite',
    'hn': 'house number'
}
```

### Context-Aware OCR Substitution Patterns
- `\b5([a-zA-Z]{3,})\b` $\to$ `s\1` (`5olutions` $\to$ `solutions`)
- `([a-zA-Z]+)5([a-zA-Z]+)` $\to$ `\1s\2`
- `\b6([a-zA-Z]{3,})\b` $\to$ `g\1` (`6reat` $\to$ `great`)
- `([a-zA-Z]+)6([a-zA-Z]+)` $\to$ `\1g\2`

---

## 4. Transliteration Approach & Known Limitations

- **Library:** `indic-transliteration` (pure local, zero external API lookups).
- **Scheme:** Transliterates Indic Unicode scripts (`devanagari`, `gujarati`, `tamil`, `bengali`, `kannada`, `telugu`, `malayalam`) into standard `ITRANS` ASCII Latin romanization.
- **Mixed Scripts:** Automatically segments hybrid strings (e.g. `"Shyam Ventures लिमिटेड"`), transliterates only non-Latin components, and preserves Latin tokens in place.
- **Known Limitations:**
  1. *Phonetic Divergence:* Transliteration is inherently approximate. For example, `"ग्रेट इंजीनियरिंग"` romanizes to `"greTa iMjIniyariMga"` rather than exact English `"great engineering"`.
  2. *Design Implication:* Transliteration is **not intended to produce exact string equality**. Its purpose is to bridge the character-set gap so that downstream n-gram similarity, fuzzy distance (Levenshtein ratio $\ge 0.50$), and embedding encoders can operate in a shared representation space.

---

## 5. Measured Full-Scale Performance

Benchmarked on local execution across all 12.5M+ rows:
- **Source 1 (2,206,821 rows):** **67.42s** (32,735 rows/sec)
- **Source 2 (5,034,616 rows):** **239.01s** (21,065 rows/sec)
- **Source 3 (5,285,603 rows):** **251.81s** (20,990 rows/sec)
- **Total 3-Source Pipeline Time (12,527,040 rows):** **558.24s (~9.3 minutes total)**, averaging **22,440 records/sec**.

Memory footprint is strictly bounded to $< 1.0\text{ GB}$ peak RSS due to single-pass unique dictionary caching and active garbage collection.

---

## 6. Items Flagged for Team Discussion

1. **Phonetic Matching for Indian Names:** Transliterated Indic names (`greta imjiniyarimga`) share substantial character overlap with Latin names (`great engineering`), but token-exact matching will fail. In Phase 3 (Similarity Features), character 3-gram and 4-gram Jaccard / Levenshtein ratio must be prioritized over exact token set equality for Indian records.
2. **Missing Address Fallback:** With ~3.35% of external records having null addresses, candidate scoring must implement a robust name-weighted fallback when `address_norm == ""` to avoid false negatives.
3. **URL/Handle Signal:** Records with `name_embedded_url` present have higher likelihood of noise in business names; downstream ML models can utilize `name_embedded_url is not None` as an informative feature.

---

## 7. Phase 2.5: Additive Phonetic Reduction Layer (`src/normalize/phonetic.py`)

### 7.1 Architecture & Selective Application
To resolve the systematic phonetic divergence between Indic-transliterated names and English reference names without altering the already-validated core normalization pipeline, an **additive, non-invasive phonetic reduction layer** is introduced.

- **Selective Activation:** Controlled by `should_apply_phonetic_reduction(script)`. The reduction runs **only** on records whose detected script is non-Latin (`devanagari`, `gujarati`, `tamil`, `telugu`, `kannada`, `bengali`, `malayalam`, `mixed`, etc.). Latin and empty records pass through unchanged.
- **DataFrame Integration:** `apply_phonetic_features(df)` adds `name_phonetic` and `name_phonetic_applied` without modifying any baseline columns.

### 7.2 Transformation Rules & Diagnostic Justifications

| Phonetic Rule | Specific Pattern | Real Diagnostic Justification (`transliteration_quality_sample.csv`) |
| :--- | :--- | :--- |
| **Regional Suffix Cleansing** | Strip unmapped regional suffix romanizations (`praivarr limirrad`, `praibheta`, `praivet`, `limirrad`, `bhiraivedh`). | In Malayalam/Tamil/Bengali/Kannada, legal suffixes are romanized with regional phonetics (e.g. `vairr midiya praivarr limirrad` vs `white media private limited`). |
| **Nasal Cluster & Assimilation** | `imga\b` $\to$ `ing`, `mga\b` $\to$ `ng`, `m(?=[consonant])` $\to$ `n`. | Indic ITRANS adds nasal markers: `marketimga` $\to$ `marketing`, `imdiyana` $\to$ `indiana`, `phainemsa` $\to$ `phainensa`. |
| **Aspirate & Fricative Mergers** | `ph` $\to$ `f`, `v` $\to$ `w`, `bh` $\to$ `b`, `dh` $\to$ `d`, `th` $\to$ `t`, `kh` $\to$ `k`, `gh` $\to$ `g`, `sh/z` $\to$ `s`, `c/q` $\to$ `k`, `x` $\to$ `ks`. | Standardizes phonetic consonant swaps: `phyuchara phudsa` $\to$ `future foods`, `vijana` $\to$ `vision`, `bombe inphoteka` $\to$ `bombay infotech`. |
| **Vowel Neutralization** | `ee/ea/ie/ei/ii` $\to$ `i`, `oo/ou/uu` $\to$ `u`, `ai/ay` $\to$ `e`, `au/aw` $\to$ `o`. | Standardizes long vs short vowel romanization variations (`istarna` $\to$ `eastern`, `phudsa` $\to$ `foods`). |
| **Duplicate Character Collapse** | `(.)\1+` $\to$ `\1` (`tt` $\to$ `t`, `pp` $\to$ `p`, etc.). | Neutralizes double consonant emphasis generated by Dravidian/Devanagari scripts. |
| **Terminal Schwa Deletion** | Drop trailing short `a` on words with length $\ge 3$ after true consonants (`(?<=[bcdfghjklmnpqrstvwxz])a\b`). | Indic scripts append inherent short `a` vowels to word-final consonants (`jaina phudsa` $\to$ `jain foods`, `shivama` $\to$ `shivam`, `rama impeksa` $\to$ `ram impex`). |

### 7.3 Validated Before vs After Performance ($N = 500$ Ground-Truth Pairs)

Validated against the 500 ground-truth cross-script sample (`scripts/validate_phonetic_module.py`):

- **SequenceMatcher Ratio:** **0.5922 $\to$ 0.7484 (+0.1562)**
- **High-Confidence Pairs ($\ge 0.60$):** **49.0% $\to$ 85.8% (+36.8% absolute increase)**
- **Failure Pairs ($< 0.30$):** **1.8% $\to$ 1.0%**
- **Character Trigram Jaccard:** **0.1411 $\to$ 0.3119 (+120.9% gain)**
- **Character Bigram Jaccard:** **0.2573 $\to$ 0.4404 (+71.1% gain)**
- **Word Token Jaccard:** **0.0530 $\to$ 0.2035 (+283.9% gain)**

### 7.4 Downstream Guidance for Phase 3 Blocking & Matching

> [!IMPORTANT]
> **Persistent Script Disparities (Tamil & Malayalam):**
> While phonetic reduction surges accuracy across Devanagari (0.768 ratio), Kannada (0.804 ratio), Telugu (0.795 ratio), and Gujarati (0.795 ratio), **Tamil** (0.502 ratio) remains more divergent due to Tamil's unique orthography where single glyphs encode multiple voiced/unvoiced stop consonants.
> 
> **Phase 3 Recommendation:**
> 1. For non-Latin records, use `phonetic_similarity(name_a, name_b, n=2)` as a primary similarity feature.
> 2. For Tamil and Malayalam records, blocking and candidate ranking **must** incorporate address, landmark, and postal code signals to achieve high recall ($F_{0.5}$).

