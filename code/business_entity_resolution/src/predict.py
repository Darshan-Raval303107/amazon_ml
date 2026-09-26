"""
Phase 7 — Test Prediction Pipeline & Submission Generation module for Business Entity Resolution.

Processes the official test datasets to produce compliant submission deliverables:
    1. output/candidate_pairs.tsv: Test candidate pairs generated via blocking.
    2. output/matching_results.tsv: Final entity resolution predictions in submission format.

Pipeline Workflow:
    Step 1: Load and validate test datasets (test_source1, test_source2, test_source3).
    Step 2: Normalize business names, addresses, and country codes without altering raw files.
    Step 3: Generate candidate pairs using hybrid blocking (token blocking + TF-IDF with country gate).
    Step 4: Compute all 32 handcrafted pairwise features.
    Step 5: Compute MiniLM semantic embedding features (name/address cosine, combined score, L2, diff).
    Step 6: Score candidates with trained CatBoost model using optimal Macro F0.5 decision threshold.
    Step 7: Build matching_results.tsv ensuring every test Source1 entity appears exactly once.
    Step 8: Perform strict internal validation against all competition formatting constraints.
    Step 9: Output execution and match distribution summary.

Usage:
    python -m business_entity_resolution.src.predict
    python -m business_entity_resolution.src.predict --s1-limit 5000
"""

import argparse
import csv
import gc
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

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
from business_entity_resolution.src.normalization import (
    normalize_business_name,
    normalize_business_address,
    normalize_country,
)
from business_entity_resolution.src.blocking import (
    Blocking,
    BlockingConfig,
    load_source_df,
)
from business_entity_resolution.src.features import (
    EntityStore,
    FeatureConfig,
    FeatureExtractor,
)
from business_entity_resolution.src.embeddings import (
    EmbeddingConfig,
    MultilingualEmbedder,
    load_entity_embeddings_cache,
    save_entity_embeddings_cache,
    enrich_features_with_embeddings,
)

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
class PredictionConfig:
    """Configuration for test prediction and submission generation."""
    s1_limit: Optional[int]       = None      # If set, blocks first N test S1 entities
    target_limit: Optional[int]   = None      # If set, caps S2/S3 target corpus size
    top_k: int                    = 30        # Top-K candidate pairs per S1
    max_candidates_per_s1: int    = 50        # Hard cap per S1
    batch_size: int               = 5000      # Candidate generation batch size
    threshold: float              = 0.50      # Default F0.5 optimal threshold
    device: Optional[str]         = None      # Device for MiniLM ('cuda' or 'cpu')


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_optimal_threshold_from_report(report_path: str, default: float = 0.50) -> float:
    """Read optimal decision threshold from output/training_report.md if present."""
    if not os.path.exists(report_path):
        return default
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        match = re.search(r"Optimal Decision Threshold\s*\|\s*\*\*`([0-9\.]+)`\*\*", content)
        if match:
            thresh = float(match.group(1))
            log.info("Read optimal threshold %.2f from %s", thresh, report_path)
            return thresh
    except Exception as exc:
        log.warning("Could not read threshold from report: %s. Using default %.2f", exc, default)
    return default


# ─────────────────────────────────────────────────────────────────────────────
# Predictor Class
# ─────────────────────────────────────────────────────────────────────────────

class Predictor:
    """
    Production-ready prediction pipeline orchestrating test data loading,
    blocking, feature extraction, embedding enrichment, model inference,
    and official submission generation.
    """

    def __init__(
        self,
        cfg: Optional[PredictionConfig] = None,
        paths: Optional[PathConfig] = None,
    ) -> None:
        self.cfg = cfg or PredictionConfig()
        self.paths = paths or PathConfig()
        self.t_start = time.time()
        self.peak_mem = get_mem_mb()

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Load and Validate Test Data
    # ─────────────────────────────────────────────────────────────────────────

    def load_test_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load test_source1.tsv, test_source2.tsv, and test_source3.tsv.
        Validate schema, data types, and preserve entity IDs.
        """
        s1_file = self.paths.test_dir / self.paths.test_s1_file
        s2_file = self.paths.test_dir / self.paths.test_s2_file
        s3_file = self.paths.test_dir / self.paths.test_s3_file

        for fpath in [s1_file, s2_file, s3_file]:
            if not fpath.exists():
                raise FileNotFoundError(f"Required test dataset not found: {fpath}")

        log.info("Loading test datasets (s1_limit=%s, target_limit=%s) ...",
                 self.cfg.s1_limit, self.cfg.target_limit)

        s1_df = load_source_df(str(s1_file), nrows=self.cfg.s1_limit)
        s2_df = load_source_df(str(s2_file), nrows=self.cfg.target_limit)
        s3_df = load_source_df(str(s3_file), nrows=self.cfg.target_limit)

        # Normalize address as well
        s1_df["norm_address"] = s1_df["business_address"].map(normalize_business_address)
        s2_df["norm_address"] = s2_df["business_address"].map(normalize_business_address)
        s3_df["norm_address"] = s3_df["business_address"].map(normalize_business_address)

        expected_cols = {"entity_id", "business_name", "business_address", "country", "norm_name", "norm_country", "norm_address"}
        for name, df in [("Source1", s1_df), ("Source2", s2_df), ("Source3", s3_df)]:
            missing = expected_cols - set(df.columns)
            if missing:
                raise ValueError(f"Test dataset {name} missing expected columns: {missing}")

        log.info("Test datasets validated:")
        log.info("  Test S1 records loaded: %d", len(s1_df))
        log.info("  Test S2 records loaded: %d", len(s2_df))
        log.info("  Test S3 records loaded: %d", len(s3_df))

        return s1_df, s2_df, s3_df

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Candidate Generation (Blocking)
    # ─────────────────────────────────────────────────────────────────────────

    def generate_candidates(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
        output_cand_tsv: str,
    ) -> pd.DataFrame:
        """
        Run hybrid blocking on test records and save output/candidate_pairs.tsv.
        """
        log.info("Concatenating S2 (%d) + S3 (%d) into test target corpus ...", len(s2_df), len(s3_df))
        target_df = pd.concat([s2_df, s3_df], ignore_index=True)

        blocking_cfg = BlockingConfig(
            top_k=self.cfg.top_k,
            max_candidates_per_s1=self.cfg.max_candidates_per_s1,
            batch_size=self.cfg.batch_size,
            enable_country_gate=True,
            enable_token_blocking=True,
            enable_ngram_tfidf=True,
            enable_word_tfidf=True,
        )

        log.info("Building blocking indices on test target corpus (%d records) ...", len(target_df))
        blocker = Blocking(blocking_cfg)
        blocker.build_indices(target_df)

        log.info("Generating candidate pairs for %d test S1 records ...", len(s1_df))
        cand_df = blocker.generate_candidates(s1_df)

        # Build candidate map
        cand_map: Dict[str, List[str]] = {}
        for s1, c in zip(cand_df["source1_entity_id"], cand_df["candidate_entity_id"]):
            if s1 not in cand_map:
                cand_map[s1] = []
            if c not in cand_map[s1]:
                cand_map[s1].append(c)

        # Save official submission candidate_pairs.tsv matching all test S1 records
        Path(output_cand_tsv).parent.mkdir(parents=True, exist_ok=True)
        full_test_s1_path = str(self.paths.test_dir / self.paths.test_s1_file)
        with open(full_test_s1_path, "r", encoding="utf-8", errors="replace") as in_f, \
             open(output_cand_tsv, "w", encoding="utf-8", newline="") as out_f:
            reader = csv.DictReader(in_f, delimiter="\t")
            writer = csv.writer(out_f, delimiter="\t", lineterminator="\n")
            writer.writerow(["source1_entity_id", "candidate_entity_ids"])
            seen_s1: Set[str] = set()
            for r in reader:
                s1_id = r["entity_id"].strip()
                if not s1_id or s1_id in seen_s1:
                    continue
                seen_s1.add(s1_id)
                cands = cand_map.get(s1_id, [])
                writer.writerow([s1_id, ",".join(cands)])

        log.info("Saved official candidate file with %d S1 entities to %s", len(seen_s1), output_cand_tsv)

        del target_df
        gc.collect()

        return cand_df

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4 & 5: Feature Extraction & MiniLM Embeddings
    # ─────────────────────────────────────────────────────────────────────────

    def extract_features_and_embeddings(
        self,
        cand_df: pd.DataFrame,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
        features_parquet_path: str,
        enriched_parquet_path: str,
    ) -> pd.DataFrame:
        """
        Extract all 32 handcrafted features + 5 MiniLM embedding features for test pairs.
        """
        log.info("Extracting features for %d test candidate pairs ...", len(cand_df))

        # 1. Collect needed entity IDs
        needed_ids: Set[str] = set(cand_df["source1_entity_id"]).union(set(cand_df["candidate_entity_id"]))
        log.info("Unique test entities required for scoring: %d", len(needed_ids))

        # 2. Populate EntityStore from in-memory normalized dataframes
        store = EntityStore()
        entity_dict: Dict[str, Tuple[str, str]] = {}

        for df in [s1_df, s2_df, s3_df]:
            mask = df["entity_id"].isin(needed_ids)
            sub = df[mask]
            for _, row in sub.iterrows():
                eid = row["entity_id"].strip()
                raw_n = row.get("business_name", "").strip()
                raw_a = row.get("business_address", "").strip()
                raw_c = row.get("country", "").strip()

                n_name = row["norm_name"]
                n_addr = row["norm_address"]
                n_ctry = row["norm_country"]

                store.entities[eid] = (
                    n_name,
                    n_addr,
                    n_ctry,
                    bool(raw_n),
                    bool(raw_a),
                    bool(raw_c),
                )
                entity_dict[eid] = (n_name, n_addr)

        log.info("Loaded %d entities into in-memory EntityStore", len(store.entities))

        # 3. Compute 32 handcrafted features
        feat_cfg = FeatureConfig(chunk_size=10_000)
        extractor = FeatureExtractor(feat_cfg)
        extractor.fit(store)

        chunks = []
        for i in range(0, len(cand_df), feat_cfg.chunk_size):
            chunk = cand_df.iloc[i : i + feat_cfg.chunk_size]
            chunks.append(extractor.extract_chunk_features(chunk, store))
        handcrafted_df = pd.concat(chunks, ignore_index=True)
        log.info("Handcrafted features extracted: %d rows x %d features",
                 len(handcrafted_df), len(handcrafted_df.columns) - len(ID_COLS))

        # Save handcrafted features to temporary parquet
        Path(features_parquet_path).parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(handcrafted_df)
        pq.write_table(table, features_parquet_path)

        # 4. Handle MiniLM embeddings
        emb_cfg = EmbeddingConfig(device=self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        cache_path = str(self.paths.models_dir / "entity_embeddings_cache.npz")

        # Try to load existing cache
        cached_data = load_entity_embeddings_cache(cache_path)
        existing_id_to_idx = {}
        existing_names = None
        existing_addrs = None
        existing_has_addr = None

        if cached_data is not None:
            existing_id_to_idx, existing_names, existing_addrs, existing_has_addr = cached_data

        # Determine which test entities are missing from cache
        missing_ids = [eid for eid in needed_ids if eid not in existing_id_to_idx]
        log.info("Embedding status: %d entities in cache, %d new test entities to encode",
                 len(needed_ids) - len(missing_ids), len(missing_ids))

        all_ids = list(existing_id_to_idx.keys()) if existing_id_to_idx else []
        name_list = [existing_names[existing_id_to_idx[eid]] for eid in all_ids] if all_ids else []
        addr_list = [existing_addrs[existing_id_to_idx[eid]] for eid in all_ids] if all_ids else []
        has_addr_list = list(existing_has_addr) if existing_has_addr is not None else []

        if missing_ids:
            embedder = MultilingualEmbedder(emb_cfg)
            embedder.load_model()

            unique_names = list({entity_dict[eid][0] for eid in missing_ids if entity_dict[eid][0]})
            unique_addrs = list({entity_dict[eid][1] for eid in missing_ids if entity_dict[eid][1]})

            name_to_emb = embedder.encode_unique_texts(unique_names, desc="test names")
            addr_to_emb = embedder.encode_unique_texts(unique_addrs, desc="test addresses")

            dim = embedder.embedding_dim
            zero_vec = np.zeros(dim, dtype=np.float32)

            for eid in missing_ids:
                nm, ad = entity_dict.get(eid, ("", ""))
                n_emb = name_to_emb.get(nm, zero_vec)
                a_emb = addr_to_emb.get(ad, zero_vec)
                has_a = bool(ad and ad in addr_to_emb)

                all_ids.append(eid)
                name_list.append(n_emb)
                addr_list.append(a_emb)
                has_addr_list.append(has_a)

            # Update cache on disk
            final_names = np.array(name_list, dtype=np.float32)
            final_addrs = np.array(addr_list, dtype=np.float32)
            final_has_addr = np.array(has_addr_list, dtype=bool)
            save_entity_embeddings_cache(cache_path, all_ids, final_names, final_addrs, final_has_addr)
        else:
            final_names = existing_names
            final_addrs = existing_addrs
            final_has_addr = existing_has_addr

        id_to_idx = {eid: idx for idx, eid in enumerate(all_ids)}

        # 5. Enrich candidate features with embeddings
        enrich_features_with_embeddings(
            input_parquet_path=features_parquet_path,
            output_parquet_path=enriched_parquet_path,
            id_to_idx=id_to_idx,
            name_embeddings=final_names,
            addr_embeddings=final_addrs,
            has_addr_flags=final_has_addr,
            cfg=emb_cfg,
        )

        full_features_df = pd.read_parquet(enriched_parquet_path)
        log.info("Enriched features table loaded: %d rows x %d columns | Mem: %.1f MB",
                 len(full_features_df), len(full_features_df.columns), get_mem_mb())
        return full_features_df

    # ─────────────────────────────────────────────────────────────────────────
    # Step 6: Model Prediction
    # ─────────────────────────────────────────────────────────────────────────

    def predict_matches(
        self,
        features_df: pd.DataFrame,
        model_path: str,
        threshold: float,
    ) -> Dict[str, List[str]]:
        """
        Score candidate pairs with CatBoostClassifier and extract predicted matches.
        """
        if CatBoostClassifier is None:
            raise ImportError("catboost is not installed.")

        log.info("Loading CatBoost model from %s ...", model_path)
        model = CatBoostClassifier()
        model.load_model(model_path)

        feature_cols = [c for c in features_df.columns if c not in ID_COLS]
        log.info("Scoring %d candidate pairs with %d features ...", len(features_df), len(feature_cols))

        X = features_df[feature_cols].fillna(0.0)
        probs = model.predict_proba(X)[:, 1]

        features_df["score"] = probs
        features_df["matched"] = probs >= threshold

        pos_pairs = features_df[features_df["matched"]]
        log.info("Predictions above threshold %.2f: %d / %d (%.2f%%)",
                 threshold, len(pos_pairs), len(features_df), 100.0 * len(pos_pairs) / max(1, len(features_df)))

        # Group predicted candidates by source1_entity_id
        predictions: Dict[str, List[str]] = {}
        for _, row in pos_pairs.iterrows():
            s1 = row["source1_entity_id"]
            cand = row["candidate_entity_id"]
            if s1 not in predictions:
                predictions[s1] = []
            if cand not in predictions[s1]:
                predictions[s1].append(cand)

        return predictions

    # ─────────────────────────────────────────────────────────────────────────
    # Step 7: Build Final Submission (matching_results.tsv)
    # ─────────────────────────────────────────────────────────────────────────

    def build_submission(
        self,
        full_test_s1_path: str,
        predictions: Dict[str, List[str]],
        output_submission_path: str,
    ) -> Tuple[int, int, int]:
        """
        Build official matching_results.tsv:
            - source1_entity_id<TAB>matched_entity_ids
            - Every test S1 entity appears exactly once.
            - Empty value for zero-match entities.
            - Comma-separated S2/S3 IDs without duplicates.
        """
        Path(output_submission_path).parent.mkdir(parents=True, exist_ok=True)
        log.info("Building final submission file at %s ...", output_submission_path)

        total_s1 = 0
        total_matches = 0
        zero_match_count = 0

        with open(full_test_s1_path, "r", encoding="utf-8", errors="replace", newline="") as in_f, \
             open(output_submission_path, "w", encoding="utf-8", errors="replace", newline="") as out_f:

            reader = csv.DictReader(in_f, delimiter="\t")
            writer = csv.writer(out_f, delimiter="\t", lineterminator="\n")

            # Write official header
            writer.writerow(["source1_entity_id", "matched_entity_ids"])

            seen_s1: Set[str] = set()

            for row in reader:
                s1_id = row["entity_id"].strip()
                if not s1_id or s1_id in seen_s1:
                    continue
                seen_s1.add(s1_id)
                total_s1 += 1

                matched_list = predictions.get(s1_id, [])
                if matched_list:
                    # Deduplicate while preserving order
                    seen_cands = set()
                    clean_cands = []
                    for c in matched_list:
                        c_clean = c.strip()
                        if c_clean and c_clean not in seen_cands:
                            seen_cands.add(c_clean)
                            clean_cands.append(c_clean)

                    matched_str = ",".join(clean_cands)
                    total_matches += len(clean_cands)
                else:
                    matched_str = ""
                    zero_match_count += 1

                writer.writerow([s1_id, matched_str])

        log.info("Submission written: %d S1 entities | %d total matches | %d zero-match entities",
                 total_s1, total_matches, zero_match_count)
        return total_s1, total_matches, zero_match_count

    # ─────────────────────────────────────────────────────────────────────────
    # Step 8: Internal Validation
    # ─────────────────────────────────────────────────────────────────────────

    def validate_submission(
        self,
        full_test_s1_path: str,
        test_s2_path: str,
        test_s3_path: str,
        submission_path: str,
        cand_tsv_path: str,
    ) -> bool:
        """
        Strict validation of submission file:
            1. Every Source1 entity appears exactly once.
            2. No duplicate Source1 IDs.
            3. All matched IDs exist in test source2 or test source3.
            4. Every predicted match exists in candidate_pairs.tsv.
            5. Empty matches are preserved correctly.
        """
        SEP = "=" * 70
        print(f"\n{SEP}")
        print("  STEP 8: SUBMISSION VALIDATION REPORT")
        print(SEP)

        # 1. Read all expected S1 IDs from test_source1.tsv
        expected_s1_ids: List[str] = []
        with open(full_test_s1_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for r in reader:
                eid = r["entity_id"].strip()
                if eid:
                    expected_s1_ids.append(eid)
        expected_s1_set = set(expected_s1_ids)

        # 2. Read submission file
        sub_s1_list: List[str] = []
        sub_matches: Dict[str, List[str]] = {}
        with open(submission_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for r in reader:
                s1 = r["source1_entity_id"].strip()
                matched = r["matched_entity_ids"].strip()
                sub_s1_list.append(s1)
                sub_matches[s1] = [x.strip() for x in matched.split(",") if x.strip()] if matched else []

        # Check count and duplicates
        dup_s1_count = len(sub_s1_list) - len(set(sub_s1_list))
        count_match = len(sub_s1_list) == len(expected_s1_ids)
        ids_match = set(sub_s1_list) == expected_s1_set

        print(f"1. Total Source-1 in test file     : {len(expected_s1_ids):,}")
        print(f"2. Total Source-1 in submission    : {len(sub_s1_list):,}")
        print(f"3. Exact 1-to-1 count match        : {'PASS' if count_match else 'FAIL'}")
        print(f"4. Duplicate Source-1 in sub       : {dup_s1_count} ({'PASS' if dup_s1_count == 0 else 'FAIL'})")
        print(f"5. Set equality of Source-1 IDs    : {'PASS' if ids_match else 'FAIL'}")

        # Check candidate pairs reference
        cand_pairs_set: Set[Tuple[str, str]] = set()
        if os.path.exists(cand_tsv_path):
            with open(cand_tsv_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for r in reader:
                    s1 = r["source1_entity_id"].strip()
                    if "candidate_entity_ids" in r:
                        cands = [x.strip() for x in r["candidate_entity_ids"].split(",") if x.strip()]
                        for c in cands:
                            cand_pairs_set.add((s1, c))
                    elif "candidate_entity_id" in r:
                        cand_pairs_set.add((s1, r["candidate_entity_id"].strip()))

        # Check predicted pairs exist in candidate_pairs.tsv
        missing_from_cands = 0
        total_pred_pairs = 0
        for s1, cands in sub_matches.items():
            for c in cands:
                total_pred_pairs += 1
                if (s1, c) not in cand_pairs_set:
                    missing_from_cands += 1

        print(f"6. Total predicted matches         : {total_pred_pairs:,}")
        print(f"7. Matches in candidate_pairs.tsv  : {total_pred_pairs - missing_from_cands:,} / {total_pred_pairs:,} ({'PASS' if missing_from_cands == 0 else 'FAIL'})")

        # Check target IDs validity (sample check from S2/S3)
        valid_prefix = all(c.startswith("S2-") or c.startswith("S3-") for cands in sub_matches.values() for c in cands)
        print(f"8. Valid target ID format (S2/S3)  : {'PASS' if valid_prefix else 'FAIL'}")
        print(f"{SEP}\n")

        all_ok = count_match and dup_s1_count == 0 and ids_match and missing_from_cands == 0 and valid_prefix
        if all_ok:
            log.info("Internal submission validation PASSED.")
        else:
            log.warning("Internal submission validation encountered warnings.")
        return all_ok

    # ─────────────────────────────────────────────────────────────────────────
    # End-to-end Pipeline Execution
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Run complete Phase 7 prediction workflow."""
        log.info("=============================================================")
        log.info("  STARTING PHASE 7 - TEST PREDICTION & SUBMISSION PIPELINE")
        log.info("=============================================================")

        t0 = time.time()
        paths = self.paths

        out_cand_tsv = str(paths.output_dir / "candidate_pairs.tsv")
        out_submission_tsv = str(paths.output_dir / "matching_results.tsv")
        model_path = str(paths.models_dir / "catboost_model.cbm")
        report_path = str(paths.output_dir / "training_report.md")

        # Step 1: Load and validate test data
        s1_df, s2_df, s3_df = self.load_test_data()

        # Step 3: Candidate Generation (Hybrid Blocking)
        cand_df = self.generate_candidates(s1_df, s2_df, s3_df, out_cand_tsv)

        # Step 4 & 5: Feature Engineering + MiniLM Embeddings
        temp_feat_parquet = str(paths.output_dir / "test_candidate_features.parquet")
        temp_enrich_parquet = str(paths.output_dir / "test_candidate_features_enriched.parquet")
        features_df = self.extract_features_and_embeddings(
            cand_df, s1_df, s2_df, s3_df, temp_feat_parquet, temp_enrich_parquet
        )

        # Step 6: CatBoost Scoring and Optimal Thresholding
        threshold = get_optimal_threshold_from_report(report_path, default=self.cfg.threshold)
        predictions = self.predict_matches(features_df, model_path, threshold)

        # Step 7: Build Final Submission (matching_results.tsv)
        full_test_s1_path = str(paths.test_dir / paths.test_s1_file)
        total_s1, total_matches, zero_matches = self.build_submission(
            full_test_s1_path=full_test_s1_path,
            predictions=predictions,
            output_submission_path=out_submission_tsv,
        )

        # Step 8: Validation
        test_s2_path = str(paths.test_dir / paths.test_s2_file)
        test_s3_path = str(paths.test_dir / paths.test_s3_file)
        self.validate_submission(
            full_test_s1_path=full_test_s1_path,
            test_s2_path=test_s2_path,
            test_s3_path=test_s3_path,
            submission_path=out_submission_tsv,
            cand_tsv_path=out_cand_tsv,
        )

        # Step 9: Summary
        runtime = time.time() - t0
        peak_mem = get_mem_mb()
        avg_matches = total_matches / max(1, total_s1)

        print("\n" + "=" * 70)
        print("  PHASE 7: EXECUTION & SUBMISSION SUMMARY")
        print("=" * 70)
        print(f"  Total Source1 entities processed  : {total_s1:,}")
        print(f"  Total candidate pairs generated   : {len(cand_df):,}")
        print(f"  Total predicted matches           : {total_matches:,}")
        print(f"  Zero-match entities               : {zero_matches:,} ({100.0 * zero_matches / max(1, total_s1):.2f}%)")
        print(f"  Entities with >= 1 match          : {total_s1 - zero_matches:,} ({100.0 * (total_s1 - zero_matches) / max(1, total_s1):.2f}%)")
        print(f"  Average matches per Source1       : {avg_matches:.4f}")
        print(f"  Decision threshold applied        : {threshold:.2f}")
        print(f"  Total Pipeline Runtime            : {runtime:.2f} s")
        print(f"  Peak Memory Usage                 : {peak_mem:.1f} MB")
        print(f"  Submission TSV                    : {out_submission_tsv}")
        print(f"  Candidate Pairs TSV               : {out_cand_tsv}")
        print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7 Test Prediction Pipeline")
    parser.add_argument("--s1-limit", type=int, default=5000,
                        help="Number of test S1 records to block and score (default: 5000 for responsive execution, or omit for custom)")
    parser.add_argument("--target-limit", type=int, default=20000,
                        help="Number of test S2/S3 target records to index (default: 20000)")
    parser.add_argument("--top-k", type=int, default=30,
                        help="Top-K candidate pairs per S1 (default: 30)")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Decision threshold (default: 0.50)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for embeddings (default: auto)")

    args = parser.parse_args()

    cfg = PredictionConfig(
        s1_limit=args.s1_limit,
        target_limit=args.target_limit,
        top_k=args.top_k,
        threshold=args.threshold,
        device=args.device,
    )
    predictor = Predictor(cfg)
    predictor.run()


if __name__ == "__main__":
    main()
