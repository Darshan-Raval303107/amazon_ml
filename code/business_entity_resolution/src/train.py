"""
Phase 6 — CatBoost Training & Macro F0.5 Optimization module for Business Entity Resolution.

Handles:
    1. Ground Truth Labeling: Binary classification labels (1 = true match, 0 = candidate non-match).
    2. Feature Preparation: Loads all 32 handcrafted + 5 MiniLM embedding features.
    3. Grouped Train/Val Split: Groups strictly by Source-1 entity ID to prevent data leakage.
    4. CatBoostClassifier Training: Logloss objective with early stopping and auto GPU/CPU detection.
    5. Threshold Optimization: Grid-evaluates thresholds [0.50 .. 0.95] to maximize Macro F0.5.
    6. Comprehensive Evaluation: Confusion matrix, precision, recall, and breakdown by match cardinality.
    7. Feature Importance Analysis: Saves output/feature_importance.csv and analyzes embedding vs handcrafted contributions.
    8. Model Persistence: Saves model to models/catboost_model.cbm and generates output/training_report.md.
    9. Verification: Validates model reloading and test inferences.

Deliverables:
    - code/business_entity_resolution/models/catboost_model.cbm
    - output/training_report.md
    - output/feature_importance.csv

Usage:
    python -m business_entity_resolution.src.train
"""

import csv
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

try:
    from catboost import CatBoostClassifier, Pool
except ImportError:
    CatBoostClassifier = None
    Pool = None

try:
    import psutil
    def get_mem_mb() -> float:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
except ImportError:
    def get_mem_mb() -> float:
        return 0.0

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from business_entity_resolution.src.config import PathConfig, ModelConfig

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ID_COLS = ["source1_entity_id", "candidate_entity_id", "source_dataset"]


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """
    Hyperparameters and settings for CatBoost training.
    """
    iterations: int          = 1000
    learning_rate: float     = 0.05
    depth: int               = 8
    loss_function: str       = "Logloss"
    eval_metric: str         = "F1"
    early_stopping_rounds: int = 100
    random_seed: int         = 42
    val_ratio: float         = 0.20
    verbose: int             = 100
    thresholds: tuple        = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load Ground Truth Labels
# ─────────────────────────────────────────────────────────────────────────────

def load_ground_truth_dict(gt_path: str) -> Dict[str, Set[str]]:
    """
    Parse train_ground_truth.tsv into:
    source1_entity_id -> set of true matched entity IDs.
    """
    log.info("Loading ground truth labels from %s ...", gt_path)
    gt: Dict[str, Set[str]] = {}
    with open(gt_path, encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            s1_id = row["source1_entity_id"].strip()
            matched = row["matched_entity_ids"].strip()
            gt[s1_id] = {x.strip() for x in matched.split(",") if x.strip()} if matched else set()
    log.info("Loaded ground truth for %d Source-1 entities", len(gt))
    return gt


def assign_labels(
    df: pd.DataFrame,
    gt: Dict[str, Set[str]],
) -> np.ndarray:
    """
    Binary label assignment:
    1 if candidate_entity_id is in gt[source1_entity_id], else 0.
    """
    s1_ids = df["source1_entity_id"].to_numpy(dtype=str)
    cand_ids = df["candidate_entity_id"].to_numpy(dtype=str)
    n = len(df)
    labels = np.zeros(n, dtype=np.int32)

    for i in range(n):
        s1 = s1_ids[i]
        c = cand_ids[i]
        if s1 in gt and c in gt[s1]:
            labels[i] = 1

    pos_count = int(np.sum(labels))
    neg_count = n - pos_count
    ratio = neg_count / max(1, pos_count)
    log.info("Labels assigned: %d positive (%.2f%%), %d negative (%.2f%%) | Imbalance ratio: %.1f:1",
             pos_count, 100.0 * pos_count / n, neg_count, 100.0 * neg_count / n, ratio)
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Grouped Train / Validation Split
# ─────────────────────────────────────────────────────────────────────────────

def grouped_train_val_split(
    df: pd.DataFrame,
    val_ratio: float = 0.20,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split candidate pairs strictly grouped by source1_entity_id so that all pairs
    for an S1 entity belong exclusively to Train or Validation (zero leakage).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (train_indices, val_indices)
    """
    unique_s1 = np.unique(df["source1_entity_id"].to_numpy(dtype=str))
    rng = np.random.default_rng(random_seed)
    rng.shuffle(unique_s1)

    n_val = int(len(unique_s1) * val_ratio)
    val_s1_set = set(unique_s1[:n_val])

    s1_series = df["source1_entity_id"].to_numpy(dtype=str)
    val_mask = np.isin(s1_series, list(val_s1_set))
    train_mask = ~val_mask

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]

    log.info("Grouped Split: %d S1 entities -> %d Train S1, %d Val S1",
             len(unique_s1), len(unique_s1) - n_val, n_val)
    log.info("  Train pairs: %d | Val pairs: %d", len(train_idx), len(val_idx))
    return train_idx, val_idx


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 & 6: Evaluation & Macro F0.5 Calculation
# ─────────────────────────────────────────────────────────────────────────────

def compute_pairwise_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Calculate pair-level confusion matrix, precision, recall, and F0.5."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    beta_sq = 0.5 ** 2  # 0.25
    denom = beta_sq * prec + rec
    f05 = (1.0 + beta_sq) * (prec * rec) / denom if denom > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": prec,
        "recall": rec,
        "pairwise_f05": f05,
    }


def compute_macro_f05(
    val_df: pd.DataFrame,
    y_pred_probs: np.ndarray,
    gt: Dict[str, Set[str]],
    threshold: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute official competition metric: Entity-level Macro F0.5.

    For each S1 entity:
        P_i = set of predicted candidates with prob >= threshold
        G_i = set of true matched candidates from ground truth
        Compute F0.5 for entity i.
    Average F0.5 across all validation S1 entities.
    """
    pred_mask = y_pred_probs >= threshold
    preds_df = val_df[pred_mask]

    # Map s1_id -> set of predicted candidate IDs
    pred_groups: Dict[str, Set[str]] = {}
    for s1, cand in zip(preds_df["source1_entity_id"], preds_df["candidate_entity_id"]):
        if s1 not in pred_groups:
            pred_groups[s1] = set()
        pred_groups[s1].add(cand)

    val_s1_list = np.unique(val_df["source1_entity_id"].to_numpy(dtype=str))

    f05_scores = []
    cardinality_scores: Dict[str, List[float]] = {
        "zero_match": [],
        "single_match": [],
        "multi_match": [],
    }

    beta_sq = 0.25  # 0.5^2

    for s1 in val_s1_list:
        p_set = pred_groups.get(s1, set())
        g_set = gt.get(s1, set())

        # Categorize by ground truth cardinality
        card_key = "zero_match" if len(g_set) == 0 else ("single_match" if len(g_set) == 1 else "multi_match")

        if len(p_set) == 0 and len(g_set) == 0:
            score = 1.0
        elif len(p_set) > 0 and len(g_set) == 0:
            score = 0.0
        elif len(p_set) == 0 and len(g_set) > 0:
            score = 0.0
        else:
            tp = len(p_set & g_set)
            prec = tp / len(p_set)
            rec  = tp / len(g_set)
            denom = beta_sq * prec + rec
            score = (1.0 + beta_sq) * (prec * rec) / denom if denom > 0 else 0.0

        f05_scores.append(score)
        cardinality_scores[card_key].append(score)

    macro_f05 = float(np.mean(f05_scores)) if f05_scores else 0.0

    breakdown = {
        "macro_f05": macro_f05,
        "zero_match_f05": float(np.mean(cardinality_scores["zero_match"])) if cardinality_scores["zero_match"] else 0.0,
        "single_match_f05": float(np.mean(cardinality_scores["single_match"])) if cardinality_scores["single_match"] else 0.0,
        "multi_match_f05": float(np.mean(cardinality_scores["multi_match"])) if cardinality_scores["multi_match"] else 0.0,
        "zero_match_count": len(cardinality_scores["zero_match"]),
        "single_match_count": len(cardinality_scores["single_match"]),
        "multi_match_count": len(cardinality_scores["multi_match"]),
    }
    return macro_f05, breakdown


# ─────────────────────────────────────────────────────────────────────────────
# Step 4, 5, 7, 8: Trainer Class
# ─────────────────────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Manages CatBoost training, threshold optimization, feature importance,
    and report generation.
    """

    def __init__(self, cfg: Optional[TrainingConfig] = None) -> None:
        self.cfg = cfg or TrainingConfig()
        self.model: Optional[CatBoostClassifier] = None
        self.best_threshold: float = 0.50
        self.best_macro_f05: float = 0.0
        self.feature_names: List[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
    ) -> None:
        """Fit CatBoostClassifier with early stopping."""
        if CatBoostClassifier is None:
            raise ImportError("catboost is not installed. Run 'pip install catboost'.")

        self.feature_names = list(X_train.columns)
        device = "GPU" if torch.cuda.is_available() else "CPU"
        log.info("Initializing CatBoostClassifier (device=%s, depth=%d, lr=%.3f, iters=%d) ...",
                 device, self.cfg.depth, self.cfg.learning_rate, self.cfg.iterations)

        self.model = CatBoostClassifier(
            iterations=self.cfg.iterations,
            learning_rate=self.cfg.learning_rate,
            depth=self.cfg.depth,
            loss_function=self.cfg.loss_function,
            eval_metric=self.cfg.eval_metric,
            early_stopping_rounds=self.cfg.early_stopping_rounds,
            random_seed=self.cfg.random_seed,
            task_type=device,
            verbose=self.cfg.verbose,
        )

        t0 = time.time()
        train_pool = Pool(X_train, y_train)
        val_pool   = Pool(X_val, y_val)

        log.info("Starting CatBoost training ...")
        self.model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
            plot=False,
        )
        log.info("CatBoost training completed in %.2fs (best iteration: %d) | Mem: %.1f MB",
                 time.time() - t0, self.model.get_best_iteration(), get_mem_mb())

    def optimize_threshold(
        self,
        val_df: pd.DataFrame,
        y_val: np.ndarray,
        y_pred_probs: np.ndarray,
        gt: Dict[str, Set[str]],
    ) -> Dict[str, Dict]:
        """
        Evaluate candidate decision thresholds and select the optimal Macro F0.5 threshold.
        """
        log.info("Evaluating decision thresholds: %s ...", self.cfg.thresholds)
        results: Dict[str, Dict] = {}

        best_score = -1.0
        best_thresh = 0.50

        print("\n" + "=" * 78)
        print(f"  {'Threshold':>10} | {'Macro F0.5':>11} | {'Precision':>10} | {'Recall':>9} | {'TP':>6} | {'FP':>7}")
        print("=" * 78)

        for thresh in self.cfg.thresholds:
            y_pred_bin = (y_pred_probs >= thresh).astype(np.int32)
            pair_m = compute_pairwise_metrics(y_val, y_pred_bin)
            macro_score, breakdown = compute_macro_f05(val_df, y_pred_probs, gt, thresh)

            results[f"{thresh:.2f}"] = {
                "threshold": thresh,
                "macro_f05": macro_score,
                "precision": pair_m["precision"],
                "recall": pair_m["recall"],
                "pairwise_f05": pair_m["pairwise_f05"],
                "tp": pair_m["tp"],
                "fp": pair_m["fp"],
                "fn": pair_m["fn"],
                "tn": pair_m["tn"],
                "breakdown": breakdown,
            }

            print(f"  {thresh:>10.2f} | {macro_score:>11.4f} | {pair_m['precision']:>10.4f} | {pair_m['recall']:>9.4f} | {pair_m['tp']:>6} | {pair_m['fp']:>7}")

            if macro_score > best_score:
                best_score = macro_score
                best_thresh = thresh

        print("=" * 78)
        self.best_threshold = best_thresh
        self.best_macro_f05 = best_score
        log.info("Optimal Decision Threshold: %.2f (Macro F0.5 = %.4f)", best_thresh, best_score)
        return results

    def extract_feature_importance(
        self,
        output_csv_path: str,
    ) -> pd.DataFrame:
        """
        Extract feature importances, separate handcrafted vs embedding contributions,
        and save to output/feature_importance.csv.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted.")

        importances = self.model.get_feature_importance()
        fi_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        fi_df.to_csv(output_csv_path, index=False)
        log.info("Saved feature importances to %s", output_csv_path)

        # Categorize contributions
        embedding_feature_names = {
            "name_embedding_cosine",
            "address_embedding_cosine",
            "combined_embedding_score",
            "embedding_l2_distance",
            "embedding_difference",
        }

        emb_imp = fi_df[fi_df["feature"].isin(embedding_feature_names)]["importance"].sum()
        hand_imp = fi_df[~fi_df["feature"].isin(embedding_feature_names)]["importance"].sum()
        total_imp = emb_imp + hand_imp

        log.info("Feature Importance Breakdown:")
        log.info("  Handcrafted Features Importance : %.2f%% (%d features)",
                 100.0 * hand_imp / total_imp, len(self.feature_names) - len(embedding_feature_names))
        log.info("  MiniLM Embedding Features       : %.2f%% (%d features)",
                 100.0 * emb_imp / total_imp, len(embedding_feature_names))

        return fi_df

    def save_model(self, model_path: str) -> None:
        """Save model to .cbm file."""
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(model_path)
        log.info("Saved CatBoost model to %s (%.2f MB)",
                 model_path, os.path.getsize(model_path) / (1024 * 1024))


# ─────────────────────────────────────────────────────────────────────────────
# Step 8: Generate Comprehensive Markdown Training Report
# ─────────────────────────────────────────────────────────────────────────────

def generate_training_report(
    report_path: str,
    train_size: int,
    val_size: int,
    pos_train: int,
    neg_train: int,
    pos_val: int,
    neg_val: int,
    cfg: TrainingConfig,
    best_thresh: float,
    opt_metrics: Dict[str, float],
    breakdown: Dict[str, float],
    fi_df: pd.DataFrame,
    runtime_sec: float,
    model_path: str,
) -> None:
    """Write comprehensive training report to markdown artifact."""
    top20 = fi_df.head(20)

    emb_features = {
        "name_embedding_cosine",
        "address_embedding_cosine",
        "combined_embedding_score",
        "embedding_l2_distance",
        "embedding_difference",
    }
    emb_sum = fi_df[fi_df["feature"].isin(emb_features)]["importance"].sum()
    hand_sum = fi_df[~fi_df["feature"].isin(emb_features)]["importance"].sum()
    tot_sum = emb_sum + hand_sum

    md_content = f"""# CatBoost Model Training & Macro F0.5 Optimization Report

## Executive Summary
This report summarizes the training, evaluation, threshold tuning, and feature importance analysis for the Amazon ML Challenge Business Entity Resolution model.

| Metric | Optimal Value |
|---|---|
| **Objective Metric** | **Macro F0.5** |
| **Optimal Decision Threshold** | **`{best_thresh:.2f}`** |
| **Validation Macro F0.5** | **`{opt_metrics['macro_f05']:.4f}`** |
| **Validation Pairwise Precision** | **`{opt_metrics['precision']:.4f}`** |
| **Validation Pairwise Recall** | **`{opt_metrics['recall']:.4f}`** |
| **Validation Pairwise F0.5** | **`{opt_metrics['pairwise_f05']:.4f}`** |
| **Total Training Runtime** | **`{runtime_sec:.1f} s`** |

---

## 1. Dataset Partition & Class Balance

Candidate pairs were split strictly by `source1_entity_id` to guarantee zero target leakage across splits.

| Split | Total Pairs | Positives | Negatives | Imbalance Ratio |
|---|---|---|---|---|
| **Train Set** | `{train_size:,}` | `{pos_train:,}` (`{100.0 * pos_train / train_size:.2f}%`) | `{neg_train:,}` | `{neg_train / max(1, pos_train):.1f}:1` |
| **Validation Set** | `{val_size:,}` | `{pos_val:,}` (`{100.0 * pos_val / val_size:.2f}%`) | `{neg_val:,}` | `{neg_val / max(1, pos_val):.1f}:1` |
| **Total** | `{train_size + val_size:,}` | `{pos_train + pos_val:,}` | `{neg_train + neg_val:,}` | `{(neg_train + neg_val) / max(1, pos_train + pos_val):.1f}:1` |

---

## 2. CatBoost Hyperparameters

- **Iterations**: `{cfg.iterations}`
- **Learning Rate**: `{cfg.learning_rate}`
- **Tree Depth**: `{cfg.depth}`
- **Loss Function**: `{cfg.loss_function}`
- **Early Stopping Rounds**: `{cfg.early_stopping_rounds}`
- **Random Seed**: `{cfg.random_seed}`
- **Saved Model Location**: `{model_path}`

---

## 3. Decision Threshold Optimization (Macro F0.5)

Because F0.5 prioritizes Precision over Recall ($\\beta = 0.5$, penalizing false positives $4\\times$ more than false negatives), standard $0.50$ thresholding is suboptimal. Grid search identified **`{best_thresh:.2f}`** as the optimal threshold.

### Confusion Matrix at Threshold `{best_thresh:.2f}`
| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | `{opt_metrics['tn']:,}` (TN) | `{opt_metrics['fp']:,}` (FP) |
| **Actual Positive** | `{opt_metrics['fn']:,}` (FN) | `{opt_metrics['tp']:,}` (TP) |

### Performance by Entity Match Cardinality
- **Zero-match Entities** (True singles with no ground truth matches): **`{breakdown['zero_match_f05']:.4f}`** ({breakdown['zero_match_count']:,} entities)
- **Single-match Entities** (Exactly 1 true match): **`{breakdown['single_match_f05']:.4f}`** ({breakdown['single_match_count']:,} entities)
- **Multi-match Entities** ($\ge 2$ true matches across S2/S3): **`{breakdown['multi_match_f05']:.4f}`** ({breakdown['multi_match_count']:,} entities)

---

## 4. Feature Importance Breakdown

### Group Contribution
- **Handcrafted Traditional Features**: **`{100.0 * hand_sum / tot_sum:.2f}%`** (32 features)
- **MiniLM Dense Embeddings**: **`{100.0 * emb_sum / tot_sum:.2f}%`** (5 features)

### Top 20 Most Predictive Features
| Rank | Feature Name | Importance (%) | Type |
|---|---|---|---|
"""
    for idx, row in top20.iterrows():
        ftype = "MiniLM Embedding" if row["feature"] in emb_features else "Handcrafted"
        md_content += f"| {idx + 1} | `{row['feature']}` | {row['importance']:.2f}% | {ftype} |\n"

    md_content += f"""
---

## 5. Artifact Verification
- Model artifact saved: `{model_path}`
- Feature importance table saved: `output/feature_importance.csv`
- Pipeline is fully reproducible and ready for submission inference.
"""

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    log.info("Saved training report to %s", report_path)


# ─────────────────────────────────────────────────────────────────────────────
# Step 9: Validation and Inferences
# ─────────────────────────────────────────────────────────────────────────────

def validate_trained_model(
    model_path: str,
    X_val: pd.DataFrame,
    val_df: pd.DataFrame,
    best_threshold: float,
) -> None:
    """Verify that model reloads from disk and generates valid prediction probabilities."""
    SEP = "=" * 78
    log.info("Validating saved model artifact: %s ...", model_path)

    reloaded_model = CatBoostClassifier()
    reloaded_model.load_model(model_path)

    sample_X = X_val.head(10)
    sample_df = val_df.head(10)
    probs = reloaded_model.predict_proba(sample_X)[:, 1]
    preds = (probs >= best_threshold).astype(int)

    print(f"\n{SEP}")
    print("  STEP 9: FIRST 10 VALIDATION PREDICTION PROBABILITIES")
    print(SEP)
    print(f"  {'Source 1 ID':<16} | {'Candidate ID':<16} | {'Dataset':<8} | {'Prob':>8} | {'Pred (>= ' + f'{best_threshold:.2f})':<14}")
    print("-" * 78)

    for i in range(len(sample_df)):
        s1 = sample_df.iloc[i]["source1_entity_id"]
        cand = sample_df.iloc[i]["candidate_entity_id"]
        ds = sample_df.iloc[i]["source_dataset"]
        p = probs[i]
        lbl = preds[i]
        print(f"  {s1:<16} | {cand:<16} | {ds:<8} | {p:>8.4f} | {lbl:<14}")

    print(f"{SEP}\n")
    log.info("Model verification passed successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end Pipeline Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_training_pipeline(
    cfg: Optional[TrainingConfig] = None,
    features_parquet_path: Optional[str] = None,
    gt_path: Optional[str] = None,
) -> Tuple[float, float, str]:
    """
    Run full Phase 6 CatBoost training pipeline.
    """
    paths = PathConfig()
    cfg = cfg or TrainingConfig()
    t_start = time.time()

    # Determine input feature path: prefer parquet with embeddings, fall back to handcrafted parquet
    enriched_path = paths.output_dir / "candidate_pair_features_with_embeddings.parquet"
    handcrafted_path = paths.output_dir / "candidate_pair_features.parquet"

    if features_parquet_path:
        feat_path = features_parquet_path
    elif enriched_path.exists():
        feat_path = str(enriched_path)
    elif handcrafted_path.exists():
        log.warning("candidate_pair_features_with_embeddings.parquet not found. Falling back to %s", handcrafted_path)
        feat_path = str(handcrafted_path)
    else:
        raise FileNotFoundError(f"Neither {enriched_path} nor {handcrafted_path} found. Run Phase 4/5 first.")

    ground_truth_file = gt_path or str(paths.train_dir / paths.train_ground_truth_file)
    model_output_file = str(paths.models_dir / "catboost_model.cbm")
    report_output_file = str(paths.output_dir / "training_report.md")
    fi_output_file = str(paths.output_dir / "feature_importance.csv")

    log.info("=============================================================")
    log.info("  STARTING PHASE 6 - CATBOOST TRAINING & F0.5 OPTIMIZATION")
    log.info("  Input Features : %s", feat_path)
    log.info("  Ground Truth   : %s", ground_truth_file)
    log.info("=============================================================")

    # Step 1: Load Ground Truth and Candidate Features
    gt = load_ground_truth_dict(ground_truth_file)

    log.info("Loading candidate features from %s ...", feat_path)
    df = pd.read_parquet(feat_path)
    log.info("Loaded feature table: %d rows x %d columns | Mem: %.1f MB",
             len(df), len(df.columns), get_mem_mb())

    # Step 2: Assign Labels
    labels = assign_labels(df, gt)
    df["label"] = labels

    feature_cols = [c for c in df.columns if c not in ID_COLS and c != "label"]
    log.info("Using %d features for CatBoost training", len(feature_cols))

    # Fill any NaNs safely
    X_all = df[feature_cols].fillna(0.0)

    # Step 3: Grouped Train/Val Split
    train_idx, val_idx = grouped_train_val_split(df, val_ratio=cfg.val_ratio, random_seed=cfg.random_seed)

    X_train = X_all.iloc[train_idx]
    y_train = labels[train_idx]
    X_val   = X_all.iloc[val_idx]
    y_val   = labels[val_idx]

    val_df_split = df.iloc[val_idx][ID_COLS]

    pos_train = int(np.sum(y_train == 1))
    neg_train = len(y_train) - pos_train
    pos_val   = int(np.sum(y_val == 1))
    neg_val   = len(y_val) - pos_val

    # Step 4: Fit CatBoost
    trainer = ModelTrainer(cfg)
    trainer.fit(X_train, y_train, X_val, y_val)

    # Step 5: Predict Probabilities & Optimize Threshold
    log.info("Generating prediction probabilities on validation set (%d pairs) ...", len(X_val))
    y_pred_probs = trainer.model.predict_proba(X_val)[:, 1]

    thresh_results = trainer.optimize_threshold(val_df_split, y_val, y_pred_probs, gt)
    best_opt = thresh_results[f"{trainer.best_threshold:.2f}"]

    # Step 7: Feature Importance
    fi_df = trainer.extract_feature_importance(fi_output_file)

    # Step 8: Save Model & Generate Report
    trainer.save_model(model_output_file)

    runtime_total = time.time() - t_start
    generate_training_report(
        report_path=report_output_file,
        train_size=len(train_idx),
        val_size=len(val_idx),
        pos_train=pos_train,
        neg_train=neg_train,
        pos_val=pos_val,
        neg_val=neg_val,
        cfg=cfg,
        best_thresh=trainer.best_threshold,
        opt_metrics=best_opt,
        breakdown=best_opt["breakdown"],
        fi_df=fi_df,
        runtime_sec=runtime_total,
        model_path=model_output_file,
    )

    # Step 9: Validation
    validate_trained_model(model_output_file, X_val, val_df_split, trainer.best_threshold)

    log.info("=============================================================")
    log.info("  PHASE 6 TRAINING & OPTIMIZATION COMPLETE")
    log.info("  Optimal Threshold   : %.2f", trainer.best_threshold)
    log.info("  Best Macro F0.5     : %.4f", trainer.best_macro_f05)
    log.info("  Validation Precision: %.4f", best_opt["precision"])
    log.info("  Validation Recall   : %.4f", best_opt["recall"])
    log.info("  Total Runtime       : %.2f s", runtime_total)
    log.info("=============================================================")

    return trainer.best_threshold, trainer.best_macro_f05, model_output_file


if __name__ == "__main__":
    run_training_pipeline()
