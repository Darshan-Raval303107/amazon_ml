# Business Entity Resolution: Comprehensive Model Evaluation Report (Phase 3 Harness)

**Evaluation Date**: 2026-09-27  
**Dataset Evaluated**: Official Amazon ML Challenge Ground Truth (`train_ground_truth.tsv`, 2,206,821 Source-1 entities)  
**Validation Partition**: 500 Group-disjoint Source-1 entities with full target alignments (79,045 candidate pairs evaluated across 11,690 target entities)  
**Evaluation Harness**: Phase 3 Anti-Leakage & Metric Hardening Protocol  

---

## 1. Executive Summary & Headline Metrics

Under the rebuilt **Phase 3 Evaluation Harness**, the **Macro $F_{0.5}$** metric is evaluated strictly at the entity level from the final post-processed output, and true blocking recall against ground truth is reported prominently alongside it. Pairwise metrics are designated as diagnostic only.

| Metric Type | Headline Benchmark Metric | Hybrid Pipeline (Production) | CatBoost Handcrafted Only | MiniLM Dense Only |
|---|---|:---:|:---:|:---:|
| **Primary Scored Metric** | **Macro $F_{0.5}$ (Entity Level)** | **`0.8783`** | `0.8512` | `0.7301` |
| **Primary Pipeline Health** | **True Blocking Recall (vs GT)** | **`99.94%` (1,691 / 1,692)** | **`99.94%` (1,691 / 1,692)** | **`99.94%` (1,691 / 1,692)** |
| **Consistency Verification** | **Two-Methods Check (Disk vs Memory)** | **PASS (`0.878304` == `0.878304`)** | **PASS** | **PASS** |

The **Hybrid Production Pipeline** achieves **0.8783 Macro $F_{0.5}$** with zero false positive merges (**100.00% diagnostic precision**, 1,247 TP vs 0 FP) across the candidate space, outperforming both standalone CatBoost and MiniLM baselines.

---

## 2. Cardinality Breakdown with Explicit Denominators

The table below breaks down performance across entity cardinality buckets, displaying the explicit entity denominator ($n$) for every bucket:

| Cardinality Category | Definition | Hybrid Pipeline $F_{0.5}$ | CatBoost Only $F_{0.5}$ | MiniLM Only $F_{0.5}$ | Entity Denominator ($n$) |
|---|---|:---:|:---:|:---:|:---:|
| **Zero-Match Entities** | True Singletons (no true matches in GT) | **`1.0000`** | `1.0000` | `1.0000` | **$n = 28$ entities** (5.60%) |
| **Single-Match Entities** | Exactly 1 true match across S2/S3 | **`0.7692`** | `0.7308` | `0.5385` | **$n = 26$ entities** (5.20%) |
| **Multi-Match Entities** | $\ge 2$ true matches across S2/S3 | **`0.8770`** | `0.8490` | `0.7244` | **$n = 446$ entities** (89.20%) |
| **Overall Macro $F_{0.5}$** | **Unweighted Average across all entities** | **`0.8783`** | **`0.8512`** | **`0.7301`** | **$n = 500$ entities** (100.0%) |

---

## 3. Comprehensive Model Comparison

| Evaluation Metric | MiniLM Only | CatBoost Handcrafted Only | Hybrid Pipeline (Production) | Notes |
|---|---|---|---|---|
| **Macro $F_{0.5}$** | `0.7301` | `0.8512` | **`0.8783`** | **Official Scored Competition Metric** |
| **True Blocking Recall** | `99.94%` | `99.94%` | **`99.94%`** | **Found 1,691 / 1,692 ground truth pairs** |
| — S1 ↔ S2 Blocking Recall | `100.00%` | `100.00%` | **`100.00%`** | 801 / 801 true S2 matches |
| — S1 ↔ S3 Blocking Recall | `99.89%` | `99.89%` | **`99.89%`** | 890 / 891 true S3 matches |
| **Optimal Decision Threshold** | `0.85` | `0.50` | **`0.50`** | S2=`0.50`, S3=`0.50`, Singleton Floor=`0.50` |
| **Consistency Check (Method A vs B)** | `PASS` | `PASS` | **`PASS`** | Disk output matches in-memory exactly ($\Delta = 0.000000$) |
| *Pairwise Precision (Diagnostic Only)* | `0.6420` | `0.9810` | **`1.0000`** | Diagnostic only — computed over generated candidates |
| *Pairwise Recall (Diagnostic Only)* | `0.5280` | `0.7180` | **`0.7374`** | Diagnostic only — does not reflect blocking recall |
| *Pairwise F1 Score (Diagnostic Only)* | `0.5794` | `0.8292` | **`0.8489`** | Diagnostic only |
| *Pairwise ROC-AUC (Diagnostic Only)* | `0.9754` | `0.9972` | **`0.9984`** | Diagnostic only |
| *Pairwise PR-AUC (Diagnostic Only)* | `0.7812` | `0.9540` | **`0.9752`** | Diagnostic only |
| **True Positives (TP - Diagnostic)** | `893` | `1,214` | **`1,247`** | Out of 1,691 candidates with positive labels |
| **False Positives (FP - Diagnostic)** | `498` | `24` | **`0`** | Zero false merges under Hybrid + Conflict Resolution |
| **False Negatives (FN - Diagnostic)** | `798` | `477` | **`444`** | Conservative F0.5 trade-off favoring extreme precision |
| **True Negatives (TN - Diagnostic)** | `76,856` | `77,330` | **`77,354`** | Out of 77,354 candidate negative pairs |
| **Candidate Pairs Evaluated** | `79,045` | `79,045` | **`79,045`** | 158.1 average candidates per S1 entity |

---

## 4. Architectural Analysis & Postmortem Context

### 4.1 What Broke in Prior Reporting
In the previous report, candidate generation was executed with an independent `nrows=20,000` slice across `train_source1.tsv`, `train_source2.tsv`, and `train_source3.tsv`. Because entity records across these databases are not aligned row-by-row, 99.6% of the true target records were omitted from the target corpus. This capped true blocking recall at **0.35%** (only 49 of 13,897 true pairs existed in the validation candidates). Because 94.5% of non-singleton entities had 0 candidate pairs, their entity F0.5 was guaranteed to be 0.0000, collapsing the overall Macro F0.5 to **0.0630** despite healthy pairwise candidate metrics (Precision 0.84, Recall 0.857, ROC-AUC 0.998).

### 4.2 How the Phase 3 Harness Prevents Recurrence
1. **Mandatory Blocking Recall Reporting**: Every evaluation now reports True Blocking Recall against ground truth right beside Macro F0.5. Any candidate truncation immediately triggers a visible failure.
2. **Cardinality Breakdown Denominators**: Entity counts ($n$) are explicitly printed for zero-match ($n=28$), single-match ($n=26$), and multi-match ($n=446$) buckets, preventing skewed singleton scores from masking multi-match failures.
3. **Automated Two-Independent-Methods Consistency Check**: Recomputes Macro F0.5 by re-reading the written `validation_matching_results.tsv` from disk and compares it to the in-memory prediction structure, asserting exact equality ($\Delta = 0.000000$).
4. **Diagnostic Pairwise Metrics**: Pairwise metrics are explicitly labeled as diagnostic only to avoid mistaking candidate-level precision for entity-level task completion.
