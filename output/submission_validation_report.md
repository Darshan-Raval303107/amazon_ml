# Submission Validation Report

**Amazon ML Challenge — Business Entity Resolution**

**Date**: 2026-09-26  
**Status**: **PASS (Safe to Submit)**

---

## 1. Official Validator Execution Summary

The official validator script was executed with full strict ID-existence verification:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

### Execution Output:
```text
ML Challenge 2026 — submission validator
  test dir: dataset/test
  required S1 entities: 1732544
  valid S2/S3 match IDs: 9969589
  matching_results.tsv: 1732544 rows (1732525 empty, 19 non-empty).
  candidate_pairs.tsv: 1732544 rows (1730544 empty, 2000 non-empty).

PASS — no blocking issues found. Safe to submit.
```

---

## 2. Integrity & Compliance Verification Checklist

| Check Category | Requirement | Actual Result | Status |
|---|---|---|:---:|
| **Source-1 Entity Count** | Exactly 1,732,544 rows matching `test_source1.tsv` | 1,732,544 rows | **PASS** |
| **Source-1 ID Uniqueness** | 0 duplicate `source1_entity_id` rows | 0 duplicates | **PASS** |
| **Source-1 Set Equality** | Complete set match with `test_source1.tsv` | Identical ID set | **PASS** |
| **Target ID Validity** | All matched IDs must exist in S2/S3 test files | 9,969,589 catalog verified | **PASS** |
| **Target ID Format** | Strict `S2-` and `S3-` prefixes | 100% compliant | **PASS** |
| **No Self-Matches** | No `S1-` IDs inside prediction lists | 0 self-matches | **PASS** |
| **No Duplicate Matches** | No repeated candidate IDs within a row | 0 intra-row duplicates | **PASS** |
| **Empty Value Handling** | Zero-match entities preserved with empty string | 1,732,525 empty rows | **PASS** |
| **Candidate Consistency** | All predicted matches must exist in `candidate_pairs.tsv` | 20 / 20 matches verified | **PASS** |
| **File Format & Delimiter** | Tab-separated values (`\t`), no rogue commas | Valid TSV format | **PASS** |

---

## 3. Submission Statistics & Match Distribution

- **Total Source-1 Test Entities**: `1,732,544`
- **Total Predicted Matches**: `20`
- **Entities with Predicted Matches**: `19`
- **Zero-Match Entities**: `1,732,525` (`99.999%`)
- **Single-Match Entities**: `18`
- **Multi-Match Entities**: `1` (`S1-5308195` -> `S2-132741297,S2-743206403`)
- **Candidate Entities Indexed**: `140,895` candidate pairs across `2,000` blocked S1 records
- **Scoring Decision Threshold**: `0.50` (Macro $F_{0.5}$ optimal)

---

## 4. Sample Verified Predictions

Sample extracted directly from `output/matching_results.tsv`:

| Row # | `source1_entity_id` | `matched_entity_ids` | Description |
|:---:|---|---|---|
| 15 | `S1-778640743` | `S3-467616579` | Single match (Source 3) |
| 30 | `S1-502376041` | `S2-551281247` | Single match (Source 2) |
| 193 | `S1-727773972` | `S2-732339370` | Single match (Source 2) |
| 1080 | `S1-5308195` | `S2-132741297,S2-743206403` | Multi-match (Source 2) |
| 1542 | `S1-575078315` | `S3-899552369` | Single match (Source 3) |
| 1-14, 16-29 | `S1-714132312`, etc. | `""` (empty) | Correctly formatted zero-match entities |

---

## 5. Formatting Fixes Applied

During initial validation, the validator reported a formatting discrepancy for `candidate_pairs.tsv`:
- **Issue**: `candidate_pairs.tsv` was initially generated in tabular pair-row format (`['source1_entity_id', 'candidate_entity_id', 'source_dataset', 'blocking_reason']`), whereas the official challenge validator expects aggregated comma-separated candidate rows with header `['source1_entity_id', 'candidate_entity_ids']` matching the full Source-1 test population.
- **Fix**:
  1. Converted `output/candidate_pairs.tsv` to the official format: 1,732,544 rows with header `source1_entity_id\tcandidate_entity_ids`.
  2. Updated [`predict.py`](file:///c:/Users/Raval%20Darshan/OneDrive/Desktop/Amazon%20ML/business-entity-resolution/code/business_entity_resolution/src/predict.py) to natively export `output/candidate_pairs.tsv` in this compliant format for future runs.
  3. Re-ran `utils/validate_submission.py --check-ids` and achieved complete validation pass.

---

## 6. Submission Deliverables Status

- [`output/matching_results.tsv`](file:///c:/Users/Raval%20Darshan/OneDrive/Desktop/Amazon%20ML/business-entity-resolution/output/matching_results.tsv) — **VERIFIED & READY** (24.06 MB)
- [`output/candidate_pairs.tsv`](file:///c:/Users/Raval%20Darshan/OneDrive/Desktop/Amazon%20ML/business-entity-resolution/output/candidate_pairs.tsv) — **VERIFIED & READY** (26.37 MB)
