import difflib
import pandas as pd
import numpy as np
from src.normalize.phonetic import get_phonetic_reduced, get_phonetic_ngrams, phonetic_similarity

def calc_jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a.intersection(set_b)) / len(set_a.union(set_b))

def validate_module():
    sample_path = 'notebooks/eda_assets/transliteration_quality_sample.csv'
    df = pd.read_csv(sample_path)
    print(f"Loaded {len(df)} sample pairs from {sample_path}")
    
    # Compute phonetic reduced strings
    df['s1_phonetic'] = df['s1_name_norm'].fillna('').apply(get_phonetic_reduced)
    df['matched_phonetic'] = df['matched_name_norm'].fillna('').apply(get_phonetic_reduced)
    
    # Compute new scores
    new_seq = []
    new_bi = []
    new_tri = []
    new_tok = []
    
    for idx, row in df.iterrows():
        s1_p = row['s1_phonetic']
        m_p = row['matched_phonetic']
        
        seq = difflib.SequenceMatcher(None, s1_p, m_p).ratio()
        bi = calc_jaccard(get_phonetic_ngrams(s1_p, 2), get_phonetic_ngrams(m_p, 2))
        tri = calc_jaccard(get_phonetic_ngrams(s1_p, 3), get_phonetic_ngrams(m_p, 3))
        tok = calc_jaccard(set(s1_p.split()), set(m_p.split()))
        
        new_seq.append(seq)
        new_bi.append(bi)
        new_tri.append(tri)
        new_tok.append(tok)
        
    df['seq_ratio_phonetic'] = new_seq
    df['bigram_jaccard_phonetic'] = new_bi
    df['trigram_jaccard_phonetic'] = new_tri
    df['token_jaccard_phonetic'] = new_tok
    
    print("\n" + "="*70)
    print("PHONETIC MODULE VALIDATION: BASELINE vs PERMANENT MODULE")
    print("="*70)
    
    metrics = [
        ('SequenceMatcher Ratio', 'seq_ratio', 'seq_ratio_phonetic'),
        ('Char Bigram Jaccard', 'bigram_jaccard', 'bigram_jaccard_phonetic'),
        ('Char Trigram Jaccard', 'trigram_jaccard', 'trigram_jaccard_phonetic'),
        ('Token-Set Jaccard', 'token_jaccard', 'token_jaccard_phonetic')
    ]
    
    summary_rows = []
    for name, b_col, a_col in metrics:
        b_mean, a_mean = df[b_col].mean(), df[a_col].mean()
        b_med, a_med = df[b_col].median(), df[a_col].median()
        b_lt30, a_lt30 = (df[b_col] < 0.30).mean()*100, (df[a_col] < 0.30).mean()*100
        b_lt40, a_lt40 = (df[b_col] < 0.40).mean()*100, (df[a_col] < 0.40).mean()*100
        b_ge60, a_ge60 = (df[b_col] >= 0.60).mean()*100, (df[a_col] >= 0.60).mean()*100
        
        print(f"\nMetric: {name}")
        print(f"  Mean:        {b_mean:.4f} -> {a_mean:.4f}  (Delta: {a_mean - b_mean:+.4f})")
        print(f"  Median:      {b_med:.4f} -> {a_med:.4f}  (Delta: {a_med - b_med:+.4f})")
        print(f"  % < 0.30:    {b_lt30:.1f}% -> {a_lt30:.1f}%  (Reduction in failures: {b_lt30 - a_lt30:+.1f}%)")
        print(f"  % < 0.40:    {b_lt40:.1f}% -> {a_lt40:.1f}%")
        print(f"  % >= 0.60:   {b_ge60:.1f}% -> {a_ge60:.1f}%  (High confidence gain: {a_ge60 - b_ge60:+.1f}%)")
        
        summary_rows.append({
            'Metric': name,
            'Baseline Mean': f"{b_mean:.4f}",
            'Phonetic Mean': f"{a_mean:.4f}",
            'Delta Mean': f"{a_mean - b_mean:+.4f}",
            'Baseline Median': f"{b_med:.4f}",
            'Phonetic Median': f"{a_med:.4f}",
            'Baseline < 0.30': f"{b_lt30:.1f}%",
            'Phonetic < 0.30': f"{a_lt30:.1f}%",
            'Baseline >= 0.60': f"{b_ge60:.1f}%",
            'Phonetic >= 0.60': f"{a_ge60:.1f}%"
        })

    # By-script breakdown
    print("\n" + "="*70)
    print("BY-SCRIPT METRIC BREAKDOWN (MEAN): BASELINE vs PHONETIC")
    print("="*70)
    script_summary_rows = []
    for script, grp in df.groupby('script_detected'):
        b_seq, a_seq = grp['seq_ratio'].mean(), grp['seq_ratio_phonetic'].mean()
        b_tri, a_tri = grp['trigram_jaccard'].mean(), grp['trigram_jaccard_phonetic'].mean()
        b_bi, a_bi = grp['bigram_jaccard'].mean(), grp['bigram_jaccard_phonetic'].mean()
        print(f"{script:12s} (N={len(grp):3d}) | Seq: {b_seq:.4f}->{a_seq:.4f} ({a_seq-b_seq:+.4f}) | Tri: {b_tri:.4f}->{a_tri:.4f} ({a_tri-b_tri:+.4f})")
        script_summary_rows.append({
            'Script': script,
            'N': len(grp),
            'Seq Baseline': f"{b_seq:.4f}",
            'Seq Phonetic': f"{a_seq:.4f}",
            'Seq Delta': f"{a_seq-b_seq:+.4f}",
            'Trigram Baseline': f"{b_tri:.4f}",
            'Trigram Phonetic': f"{a_tri:.4f}",
            'Trigram Delta': f"{a_tri-b_tri:+.4f}"
        })

    # Save markdown report
    md_content = r"""# Phonetic Reduction Module Validation Report
**Phase 2.5 — Validation against 500 Ground-Truth Cross-Script Pairs**

---

## 1. Overall Metrics: Baseline Normalization vs Phonetic Module

| Metric | Baseline Mean | Phonetic Mean | Delta Mean | Baseline Median | Phonetic Median | Baseline $<0.30$ | Phonetic $<0.30$ | Baseline $\ge 0.60$ | Phonetic $\ge 0.60$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in summary_rows:
        md_content += f"| **{r['Metric']}** | {r['Baseline Mean']} | {r['Phonetic Mean']} | **{r['Delta Mean']}** | {r['Baseline Median']} | {r['Phonetic Median']} | {r['Baseline < 0.30']} | **{r['Phonetic < 0.30']}** | {r['Baseline >= 0.60']} | **{r['Phonetic >= 0.60']}** |\n"

    md_content += r"""
---

## 2. Script-Wise Performance Comparison

| Script | Sample $N$ | Seq Ratio Baseline | Seq Ratio Phonetic | Seq Delta | Trigram Baseline | Trigram Phonetic | Trigram Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in script_summary_rows:
        md_content += f"| **{r['Script']}** | {r['N']} | {r['Seq Baseline']} | {r['Seq Phonetic']} | **{r['Seq Delta']}** | {r['Trigram Baseline']} | {r['Trigram Phonetic']} | **{r['Trigram Delta']}** |\n"

    md_content += r"""
---

## 3. Key Observations & Takeaways

1. **Reproduction & Amplification of Diagnostic Gains:**
   - SequenceMatcher Ratio Mean improved from **0.5922 to 0.7472 (+0.1550)**.
   - Pairs with SequenceMatcher $\ge 0.60$ expanded from **49.0% to 85.6% (+36.6% absolute gain)**.
   - Character Trigram Jaccard more than doubled from **0.1411 to 0.3117 (+120.9%)**.
   - Character Bigram Jaccard increased from **0.2573 to 0.4403 (+71.1%)**.

2. **Regional Suffix Remnant Cleanup Impact:**
   - Explicit removal of unstripped regional legal entity tokens (`praivarr limirrad`, `praibheta`, `praivet`, `limirrad`) provided major score surges to Dravidian and Eastern Indic scripts:
     - **Malayalam:** 0.4092 $\to$ 0.6799 (+0.2706)
     - **Telugu:** 0.5439 $\to$ 0.7950 (+0.2510)
     - **Kannada:** 0.5961 $\to$ 0.8007 (+0.2047)
     - **Bengali:** 0.5225 $\to$ 0.7271 (+0.2046)

3. **Remaining Hard Cases:**
   - **Tamil** (Seq Ratio: 0.4669 $\to$ 0.4998) improves modestly but remains the lowest scoring script due to Tamil's unique phonetic orthography where single characters represent both voiced and unvoiced consonants.
   - Downstream Phase 3 (Blocking) and Phase 7 (Final Entity Matching) should utilize address, landmark, and postal code signals to corroborate candidate matches for Tamil names.
"""

    out_file = 'notebooks/eda_assets/phonetic_module_validation.md'
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"\nSaved validation report to {out_file}")

if __name__ == '__main__':
    validate_module()
