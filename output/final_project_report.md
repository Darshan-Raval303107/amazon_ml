# Amazon ML Challenge 2026: Final Project Report

**Project**: Business Entity Resolution  
**Status**: **Production Ready & Submission Validated (PASS)**  
**Date**: September 2026  

---

## 1. Executive Implementation Summary

The Amazon ML Challenge Business Entity Resolution project delivers an end-to-end, reproducible, memory-bounded, and competition-optimized pipeline. The system solves the $O(N \times M)$ scalability bottleneck when matching millions of noisy business descriptions across three datasets (`Source1`, `Source2`, and `Source3`), scoring candidates with a hybrid feature representation and an optimized `CatBoost` gradient boosted tree model.

### Complete Phase Progression
1. **Phase 1 — Dataset Inspection (EDA)**: Analyzed null patterns, country distributions, character lengths, and ground truth match cardinalities across all train datasets (`output/dataset_analysis_report.md`).
2. **Phase 2 — Text Normalization**: Built canonicalization algorithms for legal suffixes, diacritics (NFKC), punctuation, and ISO country codes (`code/business_entity_resolution/src/normalization.py`).
3. **Phase 3 — Scalable Blocking**: Implemented token-inverted indexing with frequency caps and sparse character/word TF-IDF cosine `NearestNeighbors` with country gating (`code/business_entity_resolution/src/blocking.py`).
4. **Phase 4 — Feature Engineering**: Extracted 32 handcrafted lexical, token, address, country, and structural similarity metrics (`code/business_entity_resolution/src/features.py`).
5. **Phase 5 — MiniLM Semantic Embeddings**: Generated 384-dimensional dense semantic vectors using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, enriching candidate pairs with 5 semantic similarity metrics (`code/business_entity_resolution/src/embeddings.py`).
6. **Phase 6 — CatBoost Training & Threshold Optimization**: Trained `CatBoostClassifier` using grouped Source-1 entity splits to prevent target leakage, optimizing decision thresholds on the competition's primary metric: **Macro $F_{0.5}$** (`code/business_entity_resolution/src/train.py`).
7. **Phase 7 — Test Prediction Pipeline**: Built streaming inference pipeline scoring test candidate pairs and building official submission files (`code/business_entity_resolution/src/predict.py`).
8. **Phase 8 — Submission Validation**: Passed official challenge validation checks with zero blocking issues (`output/submission_validation_report.md`).
9. **Phase 9 — Final Documentation & Polish**: Finalized `Documentation_template.md`, root `README.md`, `.gitignore`, and project report.

---

## 2. Runtime & Resource Utilization Summary

| Stage | Operation | Runtime | Peak RAM |
|---|---|---|---|
| **Phase 1: EDA** | Streaming analysis of 12.5M train rows | 52.3 s | ~1,100 MB |
| **Phase 2: Normalization** | Rule-based vector transforms | $< 5.0$ s | $< 400$ MB |
| **Phase 3: Blocking** | Hybrid blocking over target corpus | 2.8 s | 583.8 MB |
| **Phase 4: Features** | Extracting 32 handcrafted features | 14.2 s | 628.7 MB |
| **Phase 5: Embeddings** | MiniLM encoding & vector similarity | 46.5 s | 1,721.8 MB |
| **Phase 6: CatBoost** | Training 1,000 trees with early stopping | 45.9 s | 3,866.8 MB |
| **Phase 7: Test Prediction** | Full test dataset inference & formatting | 162.1 s | 887.8 MB |
| **Phase 8: Validation** | Full catalog ID check (10M IDs) | 18.2 s | ~2,100 MB |

---

## 3. Feature Importance Summary

A total of **37 features** were utilized for classification. Handcrafted traditional features contributed **80.94%** of tree gain, while MiniLM deep embeddings contributed **19.06%**.

### Top 15 Most Predictive Features
| Rank | Feature Name | Split Importance | Category |
|:---:|---|:---:|---|
| 1 | `embedding_difference` | 14.20% | MiniLM Semantic Embedding |
| 2 | `name_length_diff` | 13.80% | Handcrafted Lexical |
| 3 | `shared_abbreviations` | 10.35% | Handcrafted Structural |
| 4 | `name_char_3gram_sim` | 6.41% | Handcrafted $N$-gram |
| 5 | `same_last_token` | 5.70% | Handcrafted Structural |
| 6 | `name_tfidf_cosine_sim` | 5.13% | Handcrafted TF-IDF |
| 7 | `address_length_diff` | 4.63% | Handcrafted Address |
| 8 | `name_token_sort_ratio` | 4.57% | Handcrafted Lexical |
| 9 | `address_tfidf_cosine_sim` | 4.34% | Handcrafted TF-IDF |
| 10 | `address_token_count_diff` | 3.95% | Handcrafted Address |
| 11 | `name_jaccard_sim` | 3.41% | Handcrafted Lexical |
| 12 | `address_char_ngram_sim` | 3.16% | Handcrafted Address |
| 13 | `shared_numeric_tokens` | 3.03% | Handcrafted Structural |
| 14 | `name_token_count_diff` | 2.79% | Handcrafted Lexical |
| 15 | `same_first_token` | 2.34% | Handcrafted Structural |

---

## 4. Model Evaluation & Optimization Summary

- **Evaluation Metric**: Macro-averaged $F_{0.5}$
- **Pairwise Precision**: **`94.74%`** (36 True Positives, 2 False Positives)
- **Pairwise Recall**: **`73.47%`** (36 True Positives, 13 False Negatives)
- **Pairwise $F_{0.5}$**: **`89.55%`**
- **Optimal Decision Threshold**: **`0.50`**
- **Zero-Match Entity Precision**: **`100.0%`** (Correctly preserved all unlinked singletons)

---

## 5. Official Submission Deliverables Status

All required submission deliverables are present, validated, and confirmed:

| File | Path | Size | Verification Status |
|---|---|---|:---:|
| **Leaderboard Matches** | `output/matching_results.tsv` | 24.06 MB | **PASS** (1,732,544 rows) |
| **Candidate Pairs** | `output/candidate_pairs.tsv` | 25.88 MB | **PASS** (1,732,544 rows) |
| **Validation Report** | `output/submission_validation_report.md` | 4.60 KB | **PASS** (Zero warnings) |
| **Training Report** | `output/training_report.md` | 4.06 KB | Complete |
| **Trained Model** | `code/business_entity_resolution/models/catboost_model.cbm` | 0.29 MB | Validated |
| **Embeddings Cache** | `code/business_entity_resolution/models/entity_embeddings_cache.npz` | 199.96 MB | Complete |
| **Competition Template** | `Documentation_template.md` | 7.42 KB | Complete |
| **Project README** | `README.md` | 5.21 KB | Complete |
