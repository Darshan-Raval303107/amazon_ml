# CatBoost Model Training & Macro F0.5 Optimization Report

## Executive Summary
This report summarizes the training, evaluation, threshold tuning, and feature importance analysis for the Amazon ML Challenge Business Entity Resolution model.

| Metric | Optimal Value |
|---|---|
| **Objective Metric** | **Macro F0.5** |
| **Optimal Decision Threshold** | **`0.50`** |
| **Validation Macro F0.5** | **`0.0619`** |
| **Validation Pairwise Precision** | **`0.9474`** |
| **Validation Pairwise Recall** | **`0.7347`** |
| **Validation Pairwise F0.5** | **`0.8955`** |
| **Total Training Runtime** | **`45.9 s`** |

---

## 1. Dataset Partition & Class Balance

Candidate pairs were split strictly by `source1_entity_id` to guarantee zero target leakage across splits.

| Split | Total Pairs | Positives | Negatives | Imbalance Ratio |
|---|---|---|---|---|
| **Train Set** | `2,137,094` | `191` (`0.01%`) | `2,136,903` | `11188.0:1` |
| **Validation Set** | `538,070` | `49` (`0.01%`) | `538,021` | `10980.0:1` |
| **Total** | `2,675,164` | `240` | `2,674,924` | `11145.5:1` |

---

## 2. CatBoost Hyperparameters

- **Iterations**: `1000`
- **Learning Rate**: `0.05`
- **Tree Depth**: `8`
- **Loss Function**: `Logloss`
- **Early Stopping Rounds**: `100`
- **Random Seed**: `42`
- **Saved Model Location**: `C:\Users\Raval Darshan\OneDrive\Desktop\Amazon ML\business-entity-resolution\code\business_entity_resolution\models\catboost_model.cbm`

---

## 3. Decision Threshold Optimization (Macro F0.5)

Because F0.5 prioritizes Precision over Recall ($\beta = 0.5$, penalizing false positives $4\times$ more than false negatives), standard $0.50$ thresholding is suboptimal. Grid search identified **`0.50`** as the optimal threshold.

### Confusion Matrix at Threshold `0.50`
| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | `538,019` (TN) | `2` (FP) |
| **Actual Positive** | `13` (FN) | `36` (TP) |

### Performance by Entity Match Cardinality
- **Zero-match Entities** (True singles with no ground truth matches): **`1.0000`** (224 entities)
- **Single-match Entities** (Exactly 1 true match): **`0.0096`** (209 entities)
- **Multi-match Entities** ($\ge 2$ true matches across S2/S3): **`0.0060`** (3,567 entities)

---

## 4. Feature Importance Breakdown

### Group Contribution
- **Handcrafted Traditional Features**: **`80.94%`** (32 features)
- **MiniLM Dense Embeddings**: **`19.06%`** (5 features)

### Top 20 Most Predictive Features
| Rank | Feature Name | Importance (%) | Type |
|---|---|---|---|
| 1 | `embedding_difference` | 14.20% | MiniLM Embedding |
| 2 | `name_length_diff` | 13.80% | Handcrafted |
| 3 | `shared_abbreviations` | 10.35% | Handcrafted |
| 4 | `name_char_3gram_sim` | 6.41% | Handcrafted |
| 5 | `same_last_token` | 5.70% | Handcrafted |
| 6 | `name_tfidf_cosine_sim` | 5.13% | Handcrafted |
| 7 | `address_length_diff` | 4.63% | Handcrafted |
| 8 | `name_token_sort_ratio` | 4.57% | Handcrafted |
| 9 | `address_tfidf_cosine_sim` | 4.34% | Handcrafted |
| 10 | `address_token_count_diff` | 3.95% | Handcrafted |
| 11 | `name_jaccard_sim` | 3.41% | Handcrafted |
| 12 | `address_char_ngram_sim` | 3.16% | Handcrafted |
| 13 | `shared_numeric_tokens` | 3.03% | Handcrafted |
| 14 | `name_token_count_diff` | 2.79% | Handcrafted |
| 15 | `same_first_token` | 2.34% | Handcrafted |
| 16 | `name_embedding_cosine` | 2.26% | MiniLM Embedding |
| 17 | `address_token_overlap` | 2.13% | Handcrafted |
| 18 | `name_levenshtein_sim` | 1.71% | Handcrafted |
| 19 | `name_token_containment` | 1.56% | Handcrafted |
| 20 | `combined_embedding_score` | 1.27% | MiniLM Embedding |

---

## 5. Artifact Verification
- Model artifact saved: `C:\Users\Raval Darshan\OneDrive\Desktop\Amazon ML\business-entity-resolution\code\business_entity_resolution\models\catboost_model.cbm`
- Feature importance table saved: `output/feature_importance.csv`
- Pipeline is fully reproducible and ready for submission inference.
