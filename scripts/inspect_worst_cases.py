import pandas as pd

df = pd.read_csv('notebooks/eda_assets/transliteration_quality_sample.csv')
worst = df.sort_values(by='trigram_jaccard').head(25)

for idx, row in worst.iterrows():
    print(f"[{row['script_detected']}] S1: '{row['s1_name_raw']}' | Match: '{row['matched_name_raw']}'")
    print(f"    S1 Norm:    '{row['s1_name_norm']}'")
    print(f"    Match Norm: '{row['matched_name_norm']}'")
    print(f"    Seq: {row['seq_ratio']:.3f} | Bi: {row['bigram_jaccard']:.3f} | Tri: {row['trigram_jaccard']:.3f}")
    print()
