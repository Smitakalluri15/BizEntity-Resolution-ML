"""
Submission Validation and Packaging Script for Amazon ML Challenge 2026.
Checks:
  1. Row counts, headers, and entity alignment
  2. Singletons formatted as empty strings
  3. Matches is a strict subset of candidate pairs
  4. Packages <team_name>_submission.zip per competition spec
"""

import os
import sys
import zipfile
import argparse
import pandas as pd


def validate_submission_files(
    test_s1_path: str,
    matching_results_path: str,
    candidate_pairs_path: str
) -> bool:
    print("=" * 80)
    print("SUBMISSION INTEGRITY AND COMPLIANCE VALIDATOR")
    print("=" * 80)

    # 1. Check file existence
    for path in [test_s1_path, matching_results_path, candidate_pairs_path]:
        if not os.path.exists(path):
            print(f"❌ ERROR: File not found: {path}")
            return False

    print(f"Loading reference test S1 entities from {test_s1_path}...")
    s1_df = pd.read_csv(test_s1_path, sep='\t', dtype=str)
    expected_s1_ids = set(s1_df['entity_id'])
    expected_count = len(expected_s1_ids)
    print(f"   Expected Source 1 entities: {expected_count:,}")

    # 2. Validate matching_results.tsv
    print(f"\nValidating {matching_results_path}...")
    with open(matching_results_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if first_line != "source1_entity_id\tmatched_entity_ids":
            print(f"❌ Header mismatch in matching_results.tsv: expected 'source1_entity_id\\tmatched_entity_ids', got '{first_line}'")
            return False

    match_df = pd.read_csv(matching_results_path, sep='\t', dtype=str, keep_default_na=False)
    if len(match_df) != expected_count:
        print(f"❌ Row count mismatch in matching_results.tsv: expected {expected_count:,}, got {len(match_df):,}")
        return False
        
    actual_match_s1_ids = set(match_df['source1_entity_id'])
    if actual_match_s1_ids != expected_s1_ids:
        missing = expected_s1_ids - actual_match_s1_ids
        print(f"❌ S1 entities mismatch in matching_results.tsv. Missing {len(missing)} entities.")
        return False
    print(f"✅ matching_results.tsv: {len(match_df):,} rows, correct header, 100% entity alignment.")

    # 3. Validate candidate_pairs.tsv
    print(f"\nValidating {candidate_pairs_path}...")
    with open(candidate_pairs_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if first_line != "source1_entity_id\tcandidate_entity_ids":
            print(f"❌ Header mismatch in candidate_pairs.tsv: expected 'source1_entity_id\\tcandidate_entity_ids', got '{first_line}'")
            return False

    cand_df = pd.read_csv(candidate_pairs_path, sep='\t', dtype=str, keep_default_na=False)
    if len(cand_df) != expected_count:
        print(f"❌ Row count mismatch in candidate_pairs.tsv: expected {expected_count:,}, got {len(cand_df):,}")
        return False

    actual_cand_s1_ids = set(cand_df['source1_entity_id'])
    if actual_cand_s1_ids != expected_s1_ids:
        missing = expected_s1_ids - actual_cand_s1_ids
        print(f"❌ S1 entities mismatch in candidate_pairs.tsv. Missing {len(missing)} entities.")
        return False
    print(f"✅ candidate_pairs.tsv: {len(cand_df):,} rows, correct header, 100% entity alignment.")

    # 4. Validate Subset Constraint: Matches must be subset of Candidate Pairs
    print("\nVerifying that matching_results is a strict subset of candidate_pairs...")
    cand_map = {}
    for _, r in cand_df.iterrows():
        c_str = str(r['candidate_entity_ids']).strip()
        cand_map[r['source1_entity_id']] = set(c.strip() for c in c_str.split(',') if c.strip())

    violations = 0
    total_matches = 0
    total_singletons = 0
    for _, r in match_df.iterrows():
        s1_id = r['source1_entity_id']
        m_str = str(r['matched_entity_ids']).strip()
        if not m_str:
            total_singletons += 1
            continue
            
        m_set = set(m.strip() for m in m_str.split(',') if m.strip())
        total_matches += len(m_set)
        c_set = cand_map.get(s1_id, set())
        
        diff = m_set - c_set
        if diff:
            violations += 1
            if violations <= 5:
                print(f"❌ Violation on {s1_id}: matched IDs {diff} not present in candidate_pairs")

    if violations > 0:
        print(f"❌ Total subset violations: {violations}")
        return False

    print(f"✅ Strict subset verification passed (0 violations across {expected_count:,} entities).")
    print(f"   Summary: {total_matches:,} matched references, {total_singletons:,} singletons ({total_singletons/expected_count*100:.2f}%).")
    return True


def package_submission_zip(
    team_name: str,
    output_dir: str,
    doc_path: str,
    zip_output_dir: str = "output"
) -> str:
    zip_name = f"{team_name}_submission.zip"
    zip_path = os.path.join(zip_output_dir, zip_name)
    os.makedirs(zip_output_dir, exist_ok=True)

    print(f"\nPackaging final submission zip: {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 1. matching_results.tsv
        m_path = os.path.join(output_dir, "matching_results.tsv")
        zf.write(m_path, arcname="matching_results.tsv")
        print(f"   + matching_results.tsv ({os.path.getsize(m_path):,} bytes)")

        # 2. candidate_pairs.tsv
        c_path = os.path.join(output_dir, "candidate_pairs.tsv")
        zf.write(c_path, arcname="candidate_pairs.tsv")
        print(f"   + candidate_pairs.tsv ({os.path.getsize(c_path):,} bytes)")

        # 3. Documentation
        zf.write(doc_path, arcname="Documentation.md")
        print(f"   + Documentation.md ({os.path.getsize(doc_path):,} bytes)")

        # 4. Code directory
        code_prefix = "code/business_entity_resolution"
        for root, dirs, files in os.walk("src"):
            for file in files:
                if file.endswith(('.py', '.md')):
                    full_path = os.path.join(root, file)
                    arc_path = os.path.join(code_prefix, full_path)
                    zf.write(full_path, arcname=arc_path)

        for root, dirs, files in os.walk("scripts"):
            for file in files:
                if file.endswith(('.py', '.md')):
                    full_path = os.path.join(root, file)
                    arc_path = os.path.join(code_prefix, full_path)
                    zf.write(full_path, arcname=arc_path)

        if os.path.exists("README.md"):
            zf.write("README.md", arcname=os.path.join(code_prefix, "README.md"))
        if os.path.exists("requirements.txt"):
            zf.write("requirements.txt", arcname=os.path.join(code_prefix, "requirements.txt"))

    print(f"\n🎉 Successfully created submission package: {zip_path} ({os.path.getsize(zip_path):,} bytes)")
    return zip_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_s1", type=str, default="dataset/test/test_source1.tsv")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--doc_path", type=str, default="Documentation_template.md")
    parser.add_argument("--team_name", type=str, default="BizEntity_Team")
    args = parser.parse_args()

    m_path = os.path.join(args.output_dir, "matching_results.tsv")
    c_path = os.path.join(args.output_dir, "candidate_pairs.tsv")

    valid = validate_submission_files(args.test_s1, m_path, c_path)
    if not valid:
        print("\n❌ Submission validation FAILED. Please fix errors before packaging.")
        sys.exit(1)

    print("\n✅ All validation criteria satisfied!")
    package_submission_zip(args.team_name, args.output_dir, args.doc_path, args.output_dir)


if __name__ == '__main__':
    main()
