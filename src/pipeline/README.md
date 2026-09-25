# Pipeline Validation Harness (`src/pipeline/`)
**Amazon ML Challenge 2026 — Phase 3 Documentation**

---

## 1. Metric Specification: Macro $F_{0.5}$

The challenge evaluation metric is **Macro $F_{0.5}$**, computed per Source-1 ($S_1$) entity and averaged across all $N$ evaluated $S_1$ entities.

### Mathematical Formulation
For each Source-1 entity $i \in \{1, \dots, N\}$:
- **True match set:** $T_i = \{s \in S_2 \cup S_3 \mid s \text{ is a true match to } S_{1, i}\}$
- **Predicted match set:** $P_i = \{s \in S_2 \cup S_3 \mid s \text{ is predicted to match } S_{1, i}\}$

$$P_i = \frac{|T_i \cap P_i|}{|P_i|}, \quad R_i = \frac{|T_i \cap P_i|}{|T_i|}$$

For $\beta = 0.5$ ($\beta^2 = 0.25$):
$$F_{0.5, i} = \frac{(1 + 0.5^2) \cdot P_i \cdot R_i}{0.5^2 \cdot P_i + R_i} = \frac{1.25 \cdot P_i \cdot R_i}{0.25 \cdot P_i + R_i} = \frac{5 \cdot P_i \cdot R_i}{P_i + 4 \cdot R_i}$$

The final challenge score is the simple arithmetic mean across all $N$ entities:
$$\text{Macro } F_{0.5} = \frac{1}{N} \sum_{i=1}^N F_{0.5, i}$$

### Critical Edge-Case Semantics

1. **Correct Singleton Prediction ($T_i = \emptyset \text{ and } P_i = \emptyset$):**
   - Scoring: $\text{Precision}_i = 1.0$, $\text{Recall}_i = 1.0$, $\mathbf{F_{0.5, i} = 1.0}$.
   - *Rationale:* The entity has no matches in $S_2/S_3$, and the model correctly predicts no match.
2. **False Positive on Singleton ($T_i = \emptyset \text{ and } P_i \neq \emptyset$):**
   - Scoring: $\text{Precision}_i = 0.0$, $\text{Recall}_i = 0.0$, $\mathbf{F_{0.5, i} = 0.0}$.
   - *Rationale:* Any false positive match attached to a true singleton is penalized with 0.0.
3. **Missed Entity ($T_i \neq \emptyset \text{ and } P_i = \emptyset$):**
   - Scoring: $\text{Precision}_i = 0.0$, $\text{Recall}_i = 0.0$, $\mathbf{F_{0.5, i} = 0.0}$.
4. **Precision Bias at $\beta = 0.5$:**
   - Because $\beta < 1.0$, **Precision is weighted 4x more heavily than Recall** in the denominator ($P_i + 4 R_i$).
   - Example verified in `tests/test_scoring.py`:
     - High Precision ($P=1.0, R=0.5$) $\to \mathbf{F_{0.5} = 0.8333}$ (vs $F_1 = 0.6667$).
     - Low Precision ($P=0.5, R=1.0$) $\to \mathbf{F_{0.5} = 0.5556}$ (vs $F_1 = 0.6667$).

---

## 2. Train / Validation Split

A single, fixed, reproducible 80/20 train/validation split has been created and serialized to disk in `data/splits/`.

- **Level of Split:** Partitioned strictly at the **$S_1$ entity level**.
- **Data Preservation:** All $S_2$ and $S_3$ records remain accessible for candidate generation and graph resolution.
- **Stratification:** Stratified by ground-truth match count buckets (0 matches, 1 match, 2+ matches).

### Split Statistics ($N = 2,206,821$ Total $S_1$ Entities)

| Split | Count | % of Total | Singletons (0) | 1 Match | 2+ Matches |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training (`train_ids.txt`)** | 1,765,458 | 80.00% | 98,596 (5.58%) | 95,324 (5.40%) | 1,571,538 (89.02%) |
| **Validation (`val_ids.txt`)** | 441,363 | 20.00% | 24,649 (5.58%) | 23,831 (5.40%) | 392,883 (89.02%) |

### Loading the Split in Subsequent Phases
```python
from src.pipeline.split import load_splits

train_s1_ids, val_s1_ids = load_splits(splits_dir="data/splits")
```

---

## 3. Trivial Baseline Sanity-Check Result

As a verification test of the scoring harness, a trivial prediction baseline predicting **empty match sets for all 441,363 validation entities** was scored:

- **Macro Precision:** **0.055847 (5.58%)**
- **Macro Recall:** **0.055847 (5.58%)**
- **Macro $F_{0.5}$:** **0.055847 (5.58%)**
- **Expected Theoretical Value:** $24,649 / 441,363 = \mathbf{0.055847}$
- **Difference:** $< 10^{-6}$

*Conclusion:* The scoring harness correctly scores singletons and penalizes non-singletons with mathematical precision.

---

## 4. How to Run Validation in Later Phases

The unified validation entry point is `run_full_validation()` in `src/pipeline/validate.py`:

```python
from src.pipeline.validate import run_full_validation
from src.pipeline.split import load_splits

_, val_s1_ids = load_splits("data/splits")

# Validate output files, verify submission format, and compute validation Macro F0.5
report = run_full_validation(
    matching_results_path="output/matching_results.tsv",
    candidate_pairs_path="output/candidate_pairs.tsv",
    test_dir="dataset/train",          # Directory containing reference source1.tsv
    ground_truth_path="dataset/train/train_ground_truth.tsv",
    entity_ids_filter=val_s1_ids       # Evaluate only on held-out validation set
)

print(f"Format Valid: {report['format_valid']}")
if report['scores']:
    print(f"Validation Macro F0.5: {report['scores']['macro_f_beta']:.4f}")
    print(f"Validation Macro Precision: {report['scores']['macro_precision']:.4f}")
    print(f"Validation Macro Recall: {report['scores']['macro_recall']:.4f}")
```
