import re
import difflib
import pandas as pd
import numpy as np

def phonetic_simplify(text: str) -> str:
    """
    Applies deterministic phonetic reduction rules tailored for Indic-English phonetic divergence:
    - Schwa / short vowel normalization
    - Common phonetic consonant mergers (ph->f, v->w, ee/ea/ie->i, oo/ou->u, z/s->s, sh->s)
    - Transliterated nasal ending normalization (mga->ng, m->n before consonants)
    - Double letter collapsing
    - Silent trailing vowels
    """
    if not text:
        return ""
    
    t = text.lower()
    
    # 1. Indic ITRANS nasal endings: 'mga' -> 'ng', 'nga' -> 'ng'
    t = re.sub(r'mga\b', 'ng', t)
    t = re.sub(r'nga\b', 'ng', t)
    t = re.sub(r'mga', 'ng', t)
    
    # 2. Phonetic consonant equivalences
    t = re.sub(r'ph', 'f', t)
    t = re.sub(r'v', 'w', t)
    t = re.sub(r'bh', 'b', t)
    t = re.sub(r'dh', 'd', t)
    t = re.sub(r'th', 't', t)
    t = re.sub(r'kh', 'k', t)
    t = re.sub(r'gh', 'g', t)
    t = re.sub(r'ch', 'c', t)
    t = re.sub(r'sh', 's', t)
    t = re.sub(r'z', 's', t)
    t = re.sub(r'c([eiy])', r's\1', t)
    t = re.sub(r'c', 'k', t)
    t = re.sub(r'q', 'k', t)
    t = re.sub(r'x', 'ks', t)
    
    # 3. Vowel mergers
    t = re.sub(r'ee|ea|ie|ei', 'i', t)
    t = re.sub(r'oo|ou', 'u', t)
    t = re.sub(r'ai|ay', 'e', t)
    t = re.sub(r'au|aw', 'o', t)
    
    # 4. Collapse consecutive duplicate characters (e.g. tt -> t, pp -> p)
    t = re.sub(r'(.)\1+', r'\1', t)
    
    # 5. Drop silent trailing short 'a' (schwa addition in Sanskrit/Hindi ITRANS)
    t = re.sub(r'(?<=[b-df-hj-np-tv-z])a\b', '', t)
    
    # Clean whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def calc_jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a.intersection(set_b)) / len(set_a.union(set_b))

def get_char_ngrams(text: str, n: int = 3) -> set:
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def run_phonetic_experiment():
    sample_path = 'notebooks/eda_assets/transliteration_quality_sample.csv'
    df = pd.read_csv(sample_path)
    print(f"Loaded {len(df)} sample pairs from {sample_path}")
    
    # Baseline scores are already in df: seq_ratio, bigram_jaccard, trigram_jaccard
    
    # Compute simplified strings
    df['s1_phonetic'] = df['s1_name_norm'].fillna('').apply(phonetic_simplify)
    df['matched_phonetic'] = df['matched_name_norm'].fillna('').apply(phonetic_simplify)
    
    # Compute new scores
    new_seq = []
    new_bi = []
    new_tri = []
    new_tok = []
    
    for idx, row in df.iterrows():
        s1_p = row['s1_phonetic']
        m_p = row['matched_phonetic']
        
        seq = difflib.SequenceMatcher(None, s1_p, m_p).ratio()
        bi = calc_jaccard(get_char_ngrams(s1_p, 2), get_char_ngrams(m_p, 2))
        tri = calc_jaccard(get_char_ngrams(s1_p, 3), get_char_ngrams(m_p, 3))
        tok = calc_jaccard(set(s1_p.split()), set(m_p.split()))
        
        new_seq.append(seq)
        new_bi.append(bi)
        new_tri.append(tri)
        new_tok.append(tok)
        
    df['seq_ratio_phonetic'] = new_seq
    df['bigram_jaccard_phonetic'] = new_bi
    df['trigram_jaccard_phonetic'] = new_tri
    df['token_jaccard_phonetic'] = new_tok
    
    print("\n" + "="*60)
    print("PHONETIC SIMPLIFICATION EXPERIMENT: BEFORE vs AFTER")
    print("="*60)
    
    metrics = [
        ('SequenceMatcher Ratio', 'seq_ratio', 'seq_ratio_phonetic'),
        ('Char Bigram Jaccard', 'bigram_jaccard', 'bigram_jaccard_phonetic'),
        ('Char Trigram Jaccard', 'trigram_jaccard', 'trigram_jaccard_phonetic'),
        ('Token-Set Jaccard', 'token_jaccard', 'token_jaccard_phonetic')
    ]
    
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

    # Script-wise comparison for Trigram Jaccard
    print("\n" + "="*60)
    print("BY-SCRIPT TRIGRAM JACCARD: BEFORE vs AFTER")
    print("="*60)
    for script, grp in df.groupby('script_detected'):
        b_m = grp['trigram_jaccard'].mean()
        a_m = grp['trigram_jaccard_phonetic'].mean()
        b_med = grp['trigram_jaccard'].median()
        a_med = grp['trigram_jaccard_phonetic'].median()
        print(f"{script:12s} (N={len(grp):3d}) | Mean: {b_m:.4f} -> {a_m:.4f} ({a_m-b_m:+.4f}) | Median: {b_med:.4f} -> {a_med:.4f}")

    # Top improvements
    df['tri_gain'] = df['trigram_jaccard_phonetic'] - df['trigram_jaccard']
    print("\n--- Top 5 Improved Examples ---")
    for idx, row in df.sort_values(by='tri_gain', ascending=False).head(5).iterrows():
        print(f"[{row['script_detected']}] Gain: {row['tri_gain']:+.3f} (Before: {row['trigram_jaccard']:.3f} -> After: {row['trigram_jaccard_phonetic']:.3f})")
        print(f"  S1 Norm:       {row['s1_name_norm']}")
        print(f"  Matched Norm:  {row['matched_name_norm']}")
        print(f"  S1 Phonetic:   {row['s1_phonetic']}")
        print(f"  Match Phonetic:{row['matched_phonetic']}")
        print("-" * 40)

if __name__ == '__main__':
    run_phonetic_experiment()
