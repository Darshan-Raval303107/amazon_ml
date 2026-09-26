"""
Feature Engineering module for Business Entity Resolution.

Extracts traditional lexical, string-distance, token, n-gram, country, and structural
similarity features for all candidate entity pairs:

    A. Business Name Features:
       - exact normalized match
       - Levenshtein similarity (ratio)
       - Levenshtein token-sort ratio
       - Jaccard similarity (word tokens)
       - token overlap count
       - token containment
       - character 3-gram Jaccard similarity
       - TF-IDF cosine similarity (character n-grams)
       - name length difference
       - token count difference

    B. Business Address Features:
       - exact normalized match
       - Levenshtein similarity
       - Jaccard similarity
       - token overlap count
       - character n-gram similarity
       - TF-IDF cosine similarity
       - address length difference
       - address token count difference

    C. Country Features:
       - country exact match
       - missing-country indicator
       - country compatibility feature

    D. Missing Value Features:
       - missing business_name (S1 / Candidate)
       - missing business_address (S1 / Candidate / Both)
       - missing country (S1 / Candidate)

    E. Structural Features:
       - same first token
       - same last token
       - shared numeric tokens (numbers, street numbers, zip codes)
       - shared abbreviations (legal and address abbreviations)

Performance & Scalability Design:
    - Pre-transforms unique entity strings using fitted TF-IDF models to avoid
      re-vectorizing millions of candidate strings.
    - Calculates row-wise sparse dot products in vectorized C operations.
    - Employs C++ accelerated edit distances via rapidfuzz.
    - Streams candidate pairs in configurable chunks (default 100,000).
    - Writes incrementally to Parquet using PyArrow ParquetWriter to keep memory bounded.

Output:
    output/candidate_pair_features.parquet

Usage:
    python -m business_entity_resolution.src.features
"""

import gc
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import psutil
    def get_mem_mb() -> float:
        """Return current resident set size (RSS) in MB."""
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
except ImportError:
    def get_mem_mb() -> float:
        return 0.0

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from business_entity_resolution.src.normalization import (
    normalize_business_name,
    normalize_business_address,
    normalize_country,
)
from business_entity_resolution.src.config import PathConfig

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Common abbreviation vocabulary for structural features
COMMON_ABBREVIATIONS: Set[str] = {
    # Legal suffixes
    "pvt", "ltd", "corp", "inc", "llc", "co", "gmbh", "sa", "llp", "plc",
    "private", "limited", "corporation", "incorporated", "company",
    # Address abbreviations
    "st", "rd", "ave", "dr", "ln", "blvd", "hwy", "ct", "pl", "ste",
    "apt", "fl", "bldg", "street", "road", "avenue", "drive", "lane",
    "boulevard", "highway", "suite", "apartment", "floor", "building",
    "mg", "nh",
}

ID_COLS = ["source1_entity_id", "candidate_entity_id", "source_dataset"]


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FeatureConfig:
    """
    Hyperparameters for the feature engineering pipeline.

    Attributes
    ----------
    chunk_size : int
        Number of candidate pairs to process per chunk.
    tfidf_max_features : int
        Maximum vocabulary size for TF-IDF vectorizers.
    tfidf_ngram_range : Tuple[int, int]
        Character n-gram range for name and address TF-IDF vectorizers.
    compression : str
        Parquet compression codec ('snappy', 'zstd', 'gzip').
    enable_validation : bool
        Whether to run post-generation validation and print statistics.
    """
    chunk_size: int         = 100_000
    tfidf_max_features: int = 50_000
    tfidf_ngram_range: tuple = (2, 4)
    compression: str        = "snappy"
    enable_validation: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Entity Store (Filtered In-Memory Cache)
# ─────────────────────────────────────────────────────────────────────────────

class EntityStore:
    """
    Loads and normalizes only the records required by the candidate pairs.
    """

    def __init__(self) -> None:
        # entity_id -> (norm_name, norm_addr, norm_country, has_name, has_addr, has_country)
        self.entities: Dict[str, Tuple[str, str, str, bool, bool, bool]] = {}

    def load_needed_entities(
        self,
        needed_ids: Set[str],
        s1_path: str,
        s2_path: str,
        s3_path: str,
        chunksize: int = 200_000,
    ) -> None:
        """
        Scan source TSVs and cache only records whose entity_id is in needed_ids.
        """
        t0 = time.time()
        log.info("Loading %d required entities across source datasets ...", len(needed_ids))

        s1_needed = {eid for eid in needed_ids if eid.startswith("S1")}
        s2_needed = {eid for eid in needed_ids if eid.startswith("S2")}
        s3_needed = {eid for eid in needed_ids if eid.startswith("S3")}

        sources = [
            (s1_path, s1_needed, "Source1"),
            (s2_path, s2_needed, "Source2"),
            (s3_path, s3_needed, "Source3"),
        ]

        for path, target_ids, label in sources:
            if not target_ids:
                continue
            loaded = 0
            log.info("  Scanning %s for %d needed IDs ...", label, len(target_ids))
            for chunk in pd.read_csv(
                path,
                sep="\t",
                dtype=str,
                keep_default_na=False,
                chunksize=chunksize,
                encoding="utf-8",
                encoding_errors="replace",
            ):
                matched = chunk[chunk["entity_id"].isin(target_ids)]
                if matched.empty:
                    continue
                for _, row in matched.iterrows():
                    eid  = row["entity_id"].strip()
                    raw_n = row.get("business_name", "").strip()
                    raw_a = row.get("business_address", "").strip()
                    raw_c = row.get("country", "").strip()

                    norm_n = normalize_business_name(raw_n)
                    norm_a = normalize_business_address(raw_a)
                    norm_c = normalize_country(raw_c)

                    self.entities[eid] = (
                        norm_n,
                        norm_a,
                        norm_c,
                        bool(raw_n),
                        bool(raw_a),
                        bool(raw_c),
                    )
                loaded += len(matched)
                if loaded >= len(target_ids):
                    break
            log.info("    -> Found and cached %d %s entities", loaded, label)

        log.info("EntityStore ready: %d total entities in %.2fs | Mem: %.1f MB",
                 len(self.entities), time.time() - t0, get_mem_mb())

    def get(self, eid: str) -> Tuple[str, str, str, bool, bool, bool]:
        """Return tuple for entity_id, or defaults if missing."""
        return self.entities.get(eid, ("", "", "", False, False, False))


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Fits global TF-IDF representations and extracts pairwise similarity features.
    """

    def __init__(self, cfg: Optional[FeatureConfig] = None) -> None:
        self.cfg = cfg or FeatureConfig()

        self._name_vec: Optional[TfidfVectorizer] = None
        self._addr_vec: Optional[TfidfVectorizer] = None

        # Pre-computed sparse matrix for all unique cached entities
        self._name_mat: Optional[sp.csr_matrix] = None
        self._addr_mat: Optional[sp.csr_matrix] = None
        self._id_to_idx: Dict[str, int] = {}

    def fit(self, store: EntityStore) -> None:
        """
        Fit TF-IDF vectorizers and pre-transform all cached entity strings.
        """
        t0 = time.time()
        log.info("Fitting TF-IDF models over %d unique entities ...", len(store.entities))

        entity_ids = list(store.entities.keys())
        self._id_to_idx = {eid: i for i, eid in enumerate(entity_ids)}

        names = [store.entities[eid][0] for eid in entity_ids]
        addrs = [store.entities[eid][1] for eid in entity_ids]

        # 1. Name TF-IDF
        self._name_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.cfg.tfidf_ngram_range,
            max_features=self.cfg.tfidf_max_features,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self._name_mat = self._name_vec.fit_transform(names)
        log.info("  Name TF-IDF matrix: %s, nnz=%d", self._name_mat.shape, self._name_mat.nnz)

        # 2. Address TF-IDF
        self._addr_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.cfg.tfidf_ngram_range,
            max_features=self.cfg.tfidf_max_features,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self._addr_mat = self._addr_vec.fit_transform(addrs)
        log.info("  Address TF-IDF matrix: %s, nnz=%d", self._addr_mat.shape, self._addr_mat.nnz)

        log.info("FeatureExtractor fit complete in %.2fs | Mem: %.1f MB",
                 time.time() - t0, get_mem_mb())

    def extract_chunk_features(
        self,
        chunk_df: pd.DataFrame,
        store: EntityStore,
    ) -> pd.DataFrame:
        """
        Compute all required features for a chunk of candidate pairs.

        Parameters
        ----------
        chunk_df : pd.DataFrame
            Must have: source1_entity_id, candidate_entity_id, source_dataset.
        store : EntityStore
            Cached entity attribute lookups.

        Returns
        -------
        pd.DataFrame
            Feature matrix with ID columns + 32 engineered features.
        """
        n = len(chunk_df)
        s1_ids = chunk_df["source1_entity_id"].to_numpy(dtype=str)
        cand_ids = chunk_df["candidate_entity_id"].to_numpy(dtype=str)
        datasets = chunk_df["source_dataset"].to_numpy(dtype=str)

        # Pre-allocate feature arrays (float32 for ML training efficiency)
        f_name_exact        = np.zeros(n, dtype=np.float32)
        f_name_lev          = np.zeros(n, dtype=np.float32)
        f_name_tok_sort     = np.zeros(n, dtype=np.float32)
        f_name_jaccard      = np.zeros(n, dtype=np.float32)
        f_name_tok_overlap  = np.zeros(n, dtype=np.float32)
        f_name_tok_contain  = np.zeros(n, dtype=np.float32)
        f_name_char3g       = np.zeros(n, dtype=np.float32)
        f_name_len_diff     = np.zeros(n, dtype=np.float32)
        f_name_tok_diff     = np.zeros(n, dtype=np.float32)

        f_addr_exact        = np.zeros(n, dtype=np.float32)
        f_addr_lev          = np.zeros(n, dtype=np.float32)
        f_addr_jaccard      = np.zeros(n, dtype=np.float32)
        f_addr_tok_overlap  = np.zeros(n, dtype=np.float32)
        f_addr_char_ng      = np.zeros(n, dtype=np.float32)
        f_addr_len_diff     = np.zeros(n, dtype=np.float32)
        f_addr_tok_diff     = np.zeros(n, dtype=np.float32)

        f_country_exact     = np.zeros(n, dtype=np.float32)
        f_country_missing   = np.zeros(n, dtype=np.float32)
        f_country_compat    = np.zeros(n, dtype=np.float32)

        f_miss_s1_name      = np.zeros(n, dtype=np.float32)
        f_miss_cand_name    = np.zeros(n, dtype=np.float32)
        f_miss_s1_addr      = np.zeros(n, dtype=np.float32)
        f_miss_cand_addr    = np.zeros(n, dtype=np.float32)
        f_miss_s1_ctry      = np.zeros(n, dtype=np.float32)
        f_miss_cand_ctry    = np.zeros(n, dtype=np.float32)
        f_miss_both_addr    = np.zeros(n, dtype=np.float32)

        f_same_first_tok    = np.zeros(n, dtype=np.float32)
        f_same_last_tok     = np.zeros(n, dtype=np.float32)
        f_shared_nums       = np.zeros(n, dtype=np.float32)
        f_shared_abbrs      = np.zeros(n, dtype=np.float32)

        idx_s1 = np.empty(n, dtype=np.int32)
        idx_cand = np.empty(n, dtype=np.int32)

        id_to_idx = self._id_to_idx

        # Fast row-by-row iteration for lexical and structural features
        for i in range(n):
            e1 = s1_ids[i]
            e2 = cand_ids[i]

            idx_s1[i]   = id_to_idx.get(e1, -1)
            idx_cand[i] = id_to_idx.get(e2, -1)

            n1, a1, c1, has_n1, has_a1, has_c1 = store.get(e1)
            n2, a2, c2, has_n2, has_a2, has_c2 = store.get(e2)

            # ── Missing value indicators ─────────────────────────────────
            f_miss_s1_name[i]   = 0.0 if has_n1 else 1.0
            f_miss_cand_name[i] = 0.0 if has_n2 else 1.0
            f_miss_s1_addr[i]   = 0.0 if has_a1 else 1.0
            f_miss_cand_addr[i] = 0.0 if has_a2 else 1.0
            f_miss_s1_ctry[i]   = 0.0 if has_c1 else 1.0
            f_miss_cand_ctry[i] = 0.0 if has_c2 else 1.0
            f_miss_both_addr[i] = 1.0 if (not has_a1 and not has_a2) else 0.0

            # ── Country features ─────────────────────────────────────────
            if c1 and c2 and c1 == c2:
                f_country_exact[i] = 1.0
            if (not c1) or (not c2):
                f_country_missing[i] = 1.0
            if (not c1) or (not c2) or (c1 == c2):
                f_country_compat[i] = 1.0

            # ── Business Name features ───────────────────────────────────
            if n1 == n2 and n1:
                f_name_exact[i] = 1.0

            f_name_len_diff[i] = abs(len(n1) - len(n2))

            if n1 and n2:
                f_name_lev[i]      = fuzz.ratio(n1, n2) / 100.0
                f_name_tok_sort[i] = fuzz.token_sort_ratio(n1, n2) / 100.0

                toks1 = n1.split()
                toks2 = n2.split()
                f_name_tok_diff[i] = abs(len(toks1) - len(toks2))

                st1 = set(toks1)
                st2 = set(toks2)
                inter = st1 & st2
                union = st1 | st2

                if union:
                    f_name_jaccard[i] = len(inter) / len(union)
                f_name_tok_overlap[i] = len(inter)

                min_len = min(len(st1), len(st2))
                if min_len > 0:
                    f_name_tok_contain[i] = len(inter) / min_len

                # Char 3-gram similarity
                if len(n1) >= 3 and len(n2) >= 3:
                    g1 = {n1[k:k+3] for k in range(len(n1)-2)}
                    g2 = {n2[k:k+3] for k in range(len(n2)-2)}
                    u_g = g1 | g2
                    if u_g:
                        f_name_char3g[i] = len(g1 & g2) / len(u_g)
                elif n1 == n2:
                    f_name_char3g[i] = 1.0

                # Structural: same first / last token
                if toks1[0] == toks2[0]:
                    f_same_first_tok[i] = 1.0
                if toks1[-1] == toks2[-1]:
                    f_same_last_tok[i] = 1.0

            # ── Business Address features ────────────────────────────────
            if a1 == a2 and a1:
                f_addr_exact[i] = 1.0

            f_addr_len_diff[i] = abs(len(a1) - len(a2))

            if a1 and a2:
                f_addr_lev[i] = fuzz.ratio(a1, a2) / 100.0

                atoks1 = a1.split()
                atoks2 = a2.split()
                f_addr_tok_diff[i] = abs(len(atoks1) - len(atoks2))

                ast1 = set(atoks1)
                ast2 = set(atoks2)
                ainter = ast1 & ast2
                aunion = ast1 | ast2
                if aunion:
                    f_addr_jaccard[i] = len(ainter) / len(aunion)
                f_addr_tok_overlap[i] = len(ainter)

                if len(a1) >= 3 and len(a2) >= 3:
                    ag1 = {a1[k:k+3] for k in range(len(a1)-2)}
                    ag2 = {a2[k:k+3] for k in range(len(a2)-2)}
                    au_g = ag1 | ag2
                    if au_g:
                        f_addr_char_ng[i] = len(ag1 & ag2) / len(au_g)
                elif a1 == a2:
                    f_addr_char_ng[i] = 1.0

            # ── Structural: shared numeric tokens & abbreviations ────────
            all_toks1 = (n1.split() + a1.split())
            all_toks2 = (n2.split() + a2.split())

            nums1 = {tok for tok in all_toks1 if any(ch.isdigit() for ch in tok)}
            nums2 = {tok for tok in all_toks2 if any(ch.isdigit() for ch in tok)}
            f_shared_nums[i] = len(nums1 & nums2)

            set_all1 = set(all_toks1)
            set_all2 = set(all_toks2)
            f_shared_abbrs[i] = len(COMMON_ABBREVIATIONS & set_all1 & set_all2)

        # ── Vectorized TF-IDF Cosine Similarities ────────────────────────
        f_name_tfidf = np.zeros(n, dtype=np.float32)
        f_addr_tfidf = np.zeros(n, dtype=np.float32)

        valid_mask = (idx_s1 >= 0) & (idx_cand >= 0)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) > 0 and self._name_mat is not None:
            s1_sub = idx_s1[valid_indices]
            cand_sub = idx_cand[valid_indices]

            # Row-wise dot product of normalized sparse vectors = cosine similarity
            m1_name = self._name_mat[s1_sub]
            m2_name = self._name_mat[cand_sub]
            f_name_tfidf[valid_indices] = np.asarray(m1_name.multiply(m2_name).sum(axis=1)).ravel()

            if self._addr_mat is not None:
                m1_addr = self._addr_mat[s1_sub]
                m2_addr = self._addr_mat[cand_sub]
                f_addr_tfidf[valid_indices] = np.asarray(m1_addr.multiply(m2_addr).sum(axis=1)).ravel()

        # ── Assemble Output DataFrame ────────────────────────────────────
        out = pd.DataFrame({
            # Identifiers
            "source1_entity_id":          s1_ids,
            "candidate_entity_id":        cand_ids,
            "source_dataset":             datasets,

            # A. Name Features
            "name_exact_match":           f_name_exact,
            "name_levenshtein_sim":       f_name_lev,
            "name_token_sort_ratio":      f_name_tok_sort,
            "name_jaccard_sim":           f_name_jaccard,
            "name_token_overlap":         f_name_tok_overlap,
            "name_token_containment":     f_name_tok_contain,
            "name_char_3gram_sim":        f_name_char3g,
            "name_tfidf_cosine_sim":      f_name_tfidf,
            "name_length_diff":           f_name_len_diff,
            "name_token_count_diff":      f_name_tok_diff,

            # B. Address Features
            "address_exact_match":        f_addr_exact,
            "address_levenshtein_sim":    f_addr_lev,
            "address_jaccard_sim":        f_addr_jaccard,
            "address_token_overlap":      f_addr_tok_overlap,
            "address_char_ngram_sim":     f_addr_char_ng,
            "address_tfidf_cosine_sim":   f_addr_tfidf,
            "address_length_diff":        f_addr_len_diff,
            "address_token_count_diff":   f_addr_tok_diff,

            # C. Country Features
            "country_exact_match":        f_country_exact,
            "country_missing_indicator":  f_country_missing,
            "country_compatible":         f_country_compat,

            # D. Missing Value Features
            "missing_s1_name":            f_miss_s1_name,
            "missing_cand_name":          f_miss_cand_name,
            "missing_s1_address":         f_miss_s1_addr,
            "missing_cand_address":       f_miss_cand_addr,
            "missing_s1_country":         f_miss_s1_ctry,
            "missing_cand_country":       f_miss_cand_ctry,
            "missing_both_addresses":     f_miss_both_addr,

            # E. Structural Features
            "same_first_token":           f_same_first_tok,
            "same_last_token":            f_same_last_tok,
            "shared_numeric_tokens":      f_shared_nums,
            "shared_abbreviations":       f_shared_abbrs,
        })

        return out


# ─────────────────────────────────────────────────────────────────────────────
# Validation Functions
# ─────────────────────────────────────────────────────────────────────────────

def validate_features(parquet_path: str, max_rows: int = 100_000) -> None:
    """
    Validate engineered features file:
    1. Check for duplicate candidate pairs.
    2. Verify no required ID columns are missing.
    3. Report missing feature values.
    4. Print feature distributions.
    5. Show correlation for numeric features.
    6. Show the first 10 engineered rows.
    """
    SEP = "=" * 70
    log.info("Validating features in %s ...", parquet_path)

    # Read table metadata and sample for stats
    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    log.info("Total rows in Parquet: %d", total_rows)

    df_sample = pd.read_parquet(parquet_path) if total_rows <= max_rows else pd.read_parquet(parquet_path).head(max_rows)

    print(f"\n{SEP}")
    print("  FEATURE VALIDATION REPORT")
    print(SEP)

    # 1. Duplicate check
    pair_series = df_sample["source1_entity_id"] + " <-> " + df_sample["candidate_entity_id"]
    n_dupes = pair_series.duplicated().sum()
    print(f"1. Duplicate candidate pairs : {n_dupes} duplicates detected (OK)" if n_dupes == 0 else f"1. Duplicate candidate pairs : {n_dupes} DUPLICATES FOUND")

    # 2. Required columns check
    missing_cols = [col for col in ID_COLS if col not in df_sample.columns]
    print(f"2. Required ID columns       : {'All present' if not missing_cols else 'MISSING: ' + str(missing_cols)}")

    # 3. Missing values check
    feature_cols = [col for col in df_sample.columns if col not in ID_COLS]
    null_counts = df_sample[feature_cols].isnull().sum()
    total_nulls = null_counts.sum()
    print(f"3. Missing feature values    : {total_nulls} null values across {len(feature_cols)} features (OK)")

    # 4. Feature distributions
    print(f"\n{SEP}")
    print("4. Feature Distributions (Sample stats):")
    print(SEP)
    desc = df_sample[feature_cols].describe().T[["mean", "std", "min", "50%", "max"]]
    desc.columns = ["Mean", "Std", "Min", "Median", "Max"]
    print(desc.to_string(float_format=lambda x: f"{x:.4f}"))

    # 5. Correlation for key numeric features
    print(f"\n{SEP}")
    print("5. Correlation Matrix for Key Similarity Features:")
    print(SEP)
    key_features = [
        "name_exact_match",
        "name_levenshtein_sim",
        "name_token_sort_ratio",
        "name_jaccard_sim",
        "name_tfidf_cosine_sim",
        "address_levenshtein_sim",
        "address_jaccard_sim",
        "address_tfidf_cosine_sim",
        "country_exact_match",
        "shared_numeric_tokens",
    ]
    key_features = [f for f in key_features if f in df_sample.columns]
    corr_matrix = df_sample[key_features].corr()
    print(corr_matrix.round(3).to_string())

    # 6. Show first 10 rows
    print(f"\n{SEP}")
    print("6. First 10 Engineered Rows (Selected Features):")
    print(SEP)
    display_cols = ID_COLS + [
        "name_levenshtein_sim",
        "name_jaccard_sim",
        "name_tfidf_cosine_sim",
        "address_levenshtein_sim",
        "country_compatible",
        "shared_numeric_tokens",
    ]
    print(df_sample[display_cols].head(10).to_string(index=False))
    print(f"{SEP}\n")


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end Feature Engineering Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_feature_engineering(
    cfg: Optional[FeatureConfig] = None,
    candidate_pairs_path: Optional[str] = None,
    output_parquet_path: Optional[str] = None,
    max_pairs: Optional[int] = None,
) -> Tuple[int, int, float, float, float]:
    """
    Execute full feature engineering pipeline:
    1. Read candidate pairs.
    2. Identify needed entity IDs and populate EntityStore.
    3. Fit FeatureExtractor TF-IDF models.
    4. Process pairs in chunks and write to Parquet incrementally.
    5. Run validation.

    Returns
    -------
    Tuple[int, int, float, float, float]
        (total_pairs, num_features, runtime_sec, peak_mem_mb, file_size_mb)
    """
    paths = PathConfig()
    cfg = cfg or FeatureConfig()
    t_start = time.time()
    peak_mem = get_mem_mb()

    cand_path = candidate_pairs_path or str(paths.output_dir / paths.candidate_pairs_file)
    out_path  = output_parquet_path or str(paths.output_dir / "candidate_pair_features.parquet")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    log.info("Starting Feature Engineering Pipeline ...")
    log.info("Candidate pairs input : %s", cand_path)
    log.info("Parquet output target : %s", out_path)

    # 1. Collect required entity IDs from candidate pairs
    log.info("Scanning candidate pairs to determine required entity IDs ...")
    needed_ids: Set[str] = set()
    total_pairs = 0

    for chunk in pd.read_csv(
        cand_path,
        sep="\t",
        usecols=["source1_entity_id", "candidate_entity_id"],
        dtype=str,
        chunksize=cfg.chunk_size,
        nrows=max_pairs,
    ):
        needed_ids.update(chunk["source1_entity_id"])
        needed_ids.update(chunk["candidate_entity_id"])
        total_pairs += len(chunk)

    log.info("Total candidate pairs to process: %d (Unique entities required: %d)",
             total_pairs, len(needed_ids))

    # 2. Populate EntityStore
    s1_path = str(paths.train_dir / paths.train_s1_file)
    s2_path = str(paths.train_dir / paths.train_s2_file)
    s3_path = str(paths.train_dir / paths.train_s3_file)

    store = EntityStore()
    store.load_needed_entities(needed_ids, s1_path, s2_path, s3_path)

    # 3. Fit FeatureExtractor
    extractor = FeatureExtractor(cfg)
    extractor.fit(store)

    # 4. Stream candidate pairs, compute features, and write incrementally to Parquet
    log.info("Processing candidate pairs and writing to Parquet in chunks of %d ...", cfg.chunk_size)
    writer: Optional[pq.ParquetWriter] = None
    processed = 0
    num_features = 0

    chunk_reader = pd.read_csv(
        cand_path,
        sep="\t",
        dtype=str,
        chunksize=cfg.chunk_size,
        nrows=max_pairs,
    )

    n_chunks = (total_pairs + cfg.chunk_size - 1) // cfg.chunk_size

    for c_idx, chunk in enumerate(chunk_reader):
        c_t0 = time.time()
        feat_df = extractor.extract_chunk_features(chunk, store)

        table = pa.Table.from_pandas(feat_df)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression=cfg.compression)
            num_features = len(feat_df.columns) - len(ID_COLS)

        writer.write_table(table)
        processed += len(feat_df)

        curr_mem = get_mem_mb()
        peak_mem = max(peak_mem, curr_mem)
        elapsed  = time.time() - t_start
        c_dur    = time.time() - c_t0

        log.info(
            "Chunk %d/%d (%.1f%%) | Processed: %d/%d | Chunk: %.2fs | Elapsed: %.1fs | Mem: %.1f MB (Peak: %.1f MB)",
            c_idx + 1,
            n_chunks,
            100.0 * processed / total_pairs,
            processed,
            total_pairs,
            c_dur,
            elapsed,
            curr_mem,
            peak_mem,
        )

    if writer is not None:
        writer.close()

    total_time = time.time() - t_start
    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)

    log.info("═════════════════════════════════════════════════════════════")
    log.info("  FEATURE ENGINEERING COMPLETE")
    log.info("  Total Candidate Pairs Processed : %d", processed)
    log.info("  Number of Features Created      : %d", num_features)
    log.info("  Total Processing Time           : %.2f s", total_time)
    log.info("  Peak Memory Usage               : %.1f MB", peak_mem)
    log.info("  Output Parquet File Size        : %.2f MB", file_size_mb)
    log.info("  Saved Location                  : %s", out_path)
    log.info("═════════════════════════════════════════════════════════════")

    # 5. Validation
    if cfg.enable_validation:
        validate_features(out_path)

    return processed, num_features, total_time, peak_mem, file_size_mb


# ─────────────────────────────────────────────────────────────────────────────
# Main Execution Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_feature_engineering()
