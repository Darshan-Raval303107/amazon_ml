# Business Entity Resolution Solution

This repository contains an end-to-end Machine Learning and Deep Learning pipeline for the Business Entity Resolution Challenge. Given multi-source business records containing noisy names, addresses, and country labels (US, India, and France in test), the system resolves records from Source 2 and Source 3 that refer to the same real-world business entity as reference entities from Source 1.

---

## 1. System Architecture & Execution Flow

The architecture follows a modular two-stage Entity Resolution paradigm:

```
[Source 1, Source 2, Source 3 TSVs]
                 │
                 ▼
     [1. Data Loader & Schema Validation]
                 │
                 ▼
     [2. Text Normalization & Cleaning]
                 │
                 ▼
     [3. Candidate Generation / Blocking] ───► [output/candidate_pairs.tsv]
                 │ (High-recall candidate pairs)
                 ▼
     [4. Hybrid Feature Engineering]
         ├── Traditional String Similarities (Levenshtein, Jaccard, Overlap, Length)
         ├── TF-IDF Cosine Similarities (Character & Word n-grams)
         ├── Country & Metadata Features
         └── Multilingual DL Embeddings (paraphrase-multilingual-MiniLM-L12-v2)
                 │
                 ▼
     [5. ML Classifier & Optimization]
         ├── CatBoostClassifier (with pairwise ranking / binary classification)
         └── Macro-averaged F_0.5 Optimal Threshold Tuning (Precision-heavy)
                 │
                 ▼
     [6. Submission Generation & Validator] ───► [output/matching_results.tsv]
```

### Key Stages:

1. **Data Loading (`data_loader.py`)**:
   - Reads `.tsv` files with explicit tab delimiter (`sep="\t"`).
   - Validates columns (`entity_id`, `business_name`, `business_address`, `country`) and ground truth mappings (`source1_entity_id`, `matched_entity_ids`).
   - Ensures no hardcoding to specific country subsets (handles open set including France).

2. **Text Normalization (`normalization.py`)**:
   - Normalizes legal entity suffixes (e.g., *Corp*, *Corporation*, *Pvt*, *Private*, *Ltd*, *Limited*).
   - Standardizes address abbreviations (e.g., *Rd* -> *Road*, *St* -> *Street*).
   - Case folding, punctuation normalization, whitespace stripping, and transliteration handling.

3. **Candidate Generation / Blocking (`blocking.py`)**:
   - Dramatically reduces the $O(N_1 \times (N_2 + N_3))$ search space while maintaining high recall ceiling.
   - Utilizes multi-index blocking (phonetic tokens, exact tokens, postal code / locality prefixes, MinHash / LSH / TF-IDF top-k).
   - Exports `output/candidate_pairs.tsv` containing the exact candidate pool fed to the classifier.

4. **Deep Learning Embeddings (`embeddings.py`)**:
   - Generates dense contextual multilingual representations using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
   - Computes cosine similarity between Source 1 and candidate S2/S3 entity representations across name, address, and combined fields.

5. **Traditional & Hybrid Feature Engineering (`features.py`)**:
   - **String Edit Distances**: Levenshtein ratio, partial ratio, token sort ratio.
   - **Set Similarities**: Jaccard similarity of token and character n-grams.
   - **Token Overlap**: Intersection over union and containment ratios for name and address.
   - **TF-IDF Vectorization**: Character n-grams and word n-grams cosine similarity.
   - **Categorical & Length Features**: Country match indicator, string length deltas, token count ratios.

6. **Model Training & F0.5 Optimization (`train.py`, `evaluate.py`)**:
   - Trains a high-performance `CatBoostClassifier`.
   - Evaluates performance using macro-averaged $F_{0.5}$:
     $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
   - Performs threshold search on validation splits to maximize macro $F_{0.5}$, effectively accounting for singleton credit (entities with no matches).

7. **Inference & Submission Pipeline (`predict.py`, `pipeline.py`)**:
   - Generates compliant TSV outputs: `matching_results.tsv` and `candidate_pairs.tsv`.
   - Validates outputs against official validator constraints (correct headers, tab separators, no S1 self-matches, complete S1 coverage).

---

## 2. Directory Structure

```
business-entity-resolution/
├── dataset/
│   ├── train/                       # train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
│   └── test/                        # test_source1.tsv, test_source2.tsv, test_source3.tsv
├── output/
│   ├── matching_results.tsv         # Final matches (leaderboard submission)
│   └── candidate_pairs.tsv          # Candidate generation / blocking set
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py
│       │   ├── config.py            # Global configuration, hyperparameters, and file paths
│       │   ├── data_loader.py       # TSV loading, integrity checks, and data structures
│       │   ├── data_analysis.py     # Exploratory data analysis, statistics, and profiling
│       │   ├── normalization.py     # String cleaning, address/legal name normalization
│       │   ├── blocking.py          # Candidate generation and index blocking strategies
│       │   ├── features.py          # Traditional string, TF-IDF, token, and length features
│       │   ├── embeddings.py        # Sentence-transformers multilingual embedding extractor
│       │   ├── train.py             # Model training (CatBoost) and hyperparameter tuning
│       │   ├── evaluate.py          # Macro-averaged F0.5 evaluation and threshold search
│       │   ├── predict.py           # Test inference and TSV export matching exact schemas
│       │   └── pipeline.py          # End-to-end orchestration CLI
│       ├── models/
│       │   └── .gitkeep             # Serialized CatBoost models and artifacts
│       ├── experiments/
│       │   └── .gitkeep             # Experiment metrics, logs, and evaluation reports
│       ├── README.md                # System documentation and instructions
│       └── requirements.txt         # Required Python dependencies
├── utils/
│   └── validate_submission.py       # Standalone submission validator
├── Documentation_template.md        # Technical documentation write-up
├── README.md                        # Root project overview
└── .gitignore                       # Git ignore configuration
```

---

## 3. Setup and Execution (When Dependencies Are Installed)

### Installation
```bash
pip install -r requirements.txt
```

### Training Pipeline
```bash
python -m src.pipeline --mode train --data-dir ../../dataset/train
```

### Inference Pipeline
```bash
python -m src.pipeline --mode predict --test-dir ../../dataset/test --output-dir ../../output
```

### Validation of Generated Outputs
```bash
python ../../utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../dataset/test
```
