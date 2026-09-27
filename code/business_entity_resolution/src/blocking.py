"""
Blocking (Candidate Generation) module for Business Entity Resolution.

This module reduces the O(S1 x (S2+S3)) comparison space — roughly
2.2M x 10.3M = 22.7 trillion pairs — into a manageable, high-recall
candidate set before feature engineering, MiniLM embedding scoring,
and CatBoost classification.

Strategies combined into a scalable hybrid pipeline:
    1. Country gating     – hard constraint: only compare records with
                            identical normalised country strings, or when at
                            least one side has a missing country value.
    2. Name-token         – frequency-capped inverted index on significant
                            unigram tokens from the normalised business name.
    3. Char n-gram TF-IDF – sublinear TF-IDF (char n-gram, word-boundary)
                            with NearestNeighbors(metric="cosine", algorithm="brute")
                            on sparse CSR matrices (no dense matrix materialised).
    4. Word TF-IDF        – sublinear TF-IDF on word n-grams with
                            NearestNeighbors(metric="cosine", algorithm="brute")
                            on sparse CSR matrices.

Memory & Scalability Design:
    - Never materialises dense similarity matrices (O(batch x corpus)).
    - Uses NearestNeighbors on sparse CSR matrices with streaming evaluation.
    - Postings lists for high-frequency stop tokens are capped to prevent
      index explosion and combinatorial blow-ups.
    - S1 entities are processed in configurable batches with per-batch
      progress reporting (batches completed, candidates, elapsed, RAM).

Public API
----------
    BlockingConfig      – configurable hyperparameters dataclass
    Blocking            – blocking class (build_indices / generate_candidates)
    load_source_df()    – read + normalise a source TSV
    load_ground_truth() – parse train_ground_truth.tsv
    run_blocking()      – end-to-end pipeline entry point

Usage
-----
    python -m business_entity_resolution.src.blocking
"""

import csv
import gc
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

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

# ── constants ─────────────────────────────────────────────────────────────────
CANDIDATE_COLS = [
    "source1_entity_id",
    "candidate_entity_id",
    "source_dataset",
    "blocking_reason",
]

REASON_TOKEN      = "name_token"
REASON_NGRAM      = "char_ngram_tfidf"
REASON_WORD       = "word_tfidf"
REASON_ADDR_TFIDF = "addr_tfidf"
REASON_PHONETIC   = "phonetic_soundex"
REASON_SAFETY_NET = "safety_net_country"


def soundex(token: str) -> str:
    """Compute standard American Soundex code for a token (Section 1.4)."""
    token = token.upper()
    if not token or not token[0].isalpha():
        return ""
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    first = token[0]
    encoded = [first]
    prev = mapping.get(first, '0')
    for ch in token[1:]:
        code = mapping.get(ch, '0')
        if code != '0' and code != prev:
            encoded.append(code)
            prev = code
        elif code == '0':
            prev = '0'
    res = ''.join(encoded)[:4]
    return res.ljust(4, '0')


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BlockingConfig:
    """
    Tunable hyperparameters for all blocking stages.

    Attributes
    ----------
    top_k : int
        Maximum candidates returned per name TF-IDF retrieval pass per S1 entity.
    top_k_addr : int
        Maximum candidates returned per address TF-IDF retrieval pass per S1 entity.
    min_token_length : int
        Minimum character length for a name token to enter the inverted index.
    max_token_frequency : int
        Maximum documents a token can appear in before it is considered a stopword
        and excluded from inverted-index retrieval.
    enable_token_blocking : bool
        Whether to run the unigram inverted-index pass.
    enable_ngram_tfidf : bool
        Whether to run char n-gram TF-IDF retrieval.
    enable_word_tfidf : bool
        Whether to run word-level TF-IDF retrieval.
    enable_address_blocking : bool
        Whether to run the address-keyed TF-IDF retrieval pass (Section 1.1).
    enable_phonetic_blocking : bool
        Whether to run the phonetic Soundex inverted-index pass (Section 1.4).
    enable_safety_net : bool
        Whether to run the safety-net pass recovering high-similarity candidates
        across dirty/mismatched country labels (Section 1.2).
    safety_net_similarity_floor : float
        Minimum cosine similarity threshold to trigger safety-net admission.
    char_ngram_range : Tuple[int, int]
        Character n-gram range for the char TF-IDF vectorizer.
    word_ngram_range : Tuple[int, int]
        Word n-gram range for the word TF-IDF vectorizer.
    tfidf_max_features : int
        Vocabulary size cap for each TF-IDF vectorizer.
    batch_size : int
        Number of S1 rows per batch during candidate generation.
    enable_country_gate : bool
        If True, candidates with incompatible countries are discarded (unless saved by safety net).
    max_candidates_per_s1 : int
        Hard cap on total candidates per S1 entity across all methods.
    n_jobs : int
        Number of parallel jobs for NearestNeighbors (-1 for all CPU cores).
    random_seed : int
        Random seed for reproducibility.
    """
    top_k: int                       = 50
    top_k_addr: int                  = 20
    min_token_length: int            = 3
    max_token_frequency: int         = 10_000
    enable_token_blocking: bool      = True
    enable_ngram_tfidf: bool         = True
    enable_word_tfidf: bool          = True
    enable_address_blocking: bool    = True
    enable_phonetic_blocking: bool   = True
    enable_safety_net: bool          = True
    safety_net_similarity_floor: float = 0.75
    char_ngram_range: tuple          = (2, 4)
    word_ngram_range: tuple          = (1, 2)
    tfidf_max_features: int          = 200_000
    batch_size: int                  = 10_000
    enable_country_gate: bool        = True
    max_candidates_per_s1: int       = 100
    n_jobs: int                      = -1
    random_seed: int                 = 42


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_source_df(path: str, *, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load a source TSV (source1 / source2 / source3) and add normalised columns.

    Parameters
    ----------
    path : str
        Path to the TSV file.
    nrows : int, optional
        If given, only the first *nrows* data rows are read.

    Returns
    -------
    pd.DataFrame
        Original columns plus ``norm_name`` and ``norm_country``.
    """
    log.info("Loading %s ...", path)
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        nrows=nrows,
        keep_default_na=False,
        encoding="utf-8",
        encoding_errors="replace",
    )
    for col in ["entity_id", "business_name", "business_address", "country"]:
        if col not in df.columns:
            df[col] = ""

    df["norm_name"]    = df["business_name"].map(normalize_business_name)
    df["norm_address"] = df["business_address"].map(normalize_business_address)
    df["norm_country"] = df["country"].map(normalize_country)
    log.info("  -> %d rows loaded from %s", len(df), Path(path).name)
    return df


def load_ground_truth(path: str) -> Dict[str, Set[str]]:
    """
    Parse train_ground_truth.tsv into a dict mapping source1_entity_id ->
    set of matched entity_ids (from S2 and S3).

    Parameters
    ----------
    path : str

    Returns
    -------
    Dict[str, Set[str]]
    """
    gt: Dict[str, Set[str]] = {}
    log.info("Loading ground truth from %s ...", path)
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            s1_id   = row["source1_entity_id"].strip()
            ids_raw = row["matched_entity_ids"].strip()
            gt[s1_id] = (
                {x.strip() for x in ids_raw.split(",") if x.strip()}
                if ids_raw else set()
            )
    log.info("  -> %d S1 entities in ground truth", len(gt))
    return gt


# ─────────────────────────────────────────────────────────────────────────────
# Blocking class
# ─────────────────────────────────────────────────────────────────────────────

class Blocking:
    """
    Multi-strategy scalable candidate generator.

    Workflow
    --------
    1. ``build_indices(target_df)``   – fit indices on S2 ∪ S3.
    2. ``generate_candidates(s1_df)`` – query indices for each S1 entity in batches.
    3. ``save_candidates(df, path)``  – write ``candidate_pairs.tsv``.
    4. ``evaluate(...)``              – measure recall / reduction ratio.
    """

    def __init__(self, cfg: Optional[BlockingConfig] = None) -> None:
        self.cfg = cfg or BlockingConfig()

        # ── token inverted index ──────────────────────────────────────────
        # token -> list[row_position]
        self._token_index: Dict[str, List[int]] = {}

        # ── phonetic inverted index (Soundex on first 1-2 tokens) ────────
        self._phonetic_index: Dict[str, List[int]] = {}

        # ── TF-IDF engines (NearestNeighbors on CSR matrices) ─────────────
        self._ng_vec:   Optional[TfidfVectorizer]  = None
        self._ng_mat:   Optional[sp.csr_matrix]    = None
        self._ng_nn:    Optional[NearestNeighbors] = None

        self._word_vec: Optional[TfidfVectorizer]  = None
        self._word_mat: Optional[sp.csr_matrix]    = None
        self._word_nn:  Optional[NearestNeighbors] = None

        # ── Address TF-IDF engine (Section 1.1) ───────────────────────────
        self._addr_vec: Optional[TfidfVectorizer]  = None
        self._addr_mat: Optional[sp.csr_matrix]    = None
        self._addr_nn:  Optional[NearestNeighbors] = None

        # ── target snapshot (arrays for O(1) lookup) ──────────────────────
        self._target_ids:     Optional[np.ndarray] = None
        self._target_src:     Optional[np.ndarray] = None
        self._target_country: Optional[np.ndarray] = None
        self._target_address: Optional[np.ndarray] = None

        # Effective k for NN query (retrieve extra candidates to allow country filtering)
        self._nn_query_k: int = self.cfg.top_k * 2 if self.cfg.enable_country_gate else self.cfg.top_k

    # ── index building ────────────────────────────────────────────────────

    def build_indices(self, target_df: pd.DataFrame) -> None:
        """
        Build all blocking indices from a combined S2 ∪ S3 DataFrame.

        Parameters
        ----------
        target_df : pd.DataFrame
            Must contain columns: entity_id, norm_name, norm_country, norm_address.
        """
        t0 = time.time()
        n  = len(target_df)
        log.info("Building blocking indices over %d target records ...", n)
        log.info("Current Memory: %.1f MB", get_mem_mb())

        self._target_ids     = target_df["entity_id"].to_numpy(dtype=str)
        self._target_country = target_df["norm_country"].to_numpy(dtype=str)
        self._target_src     = np.array(
            ["S2" if eid.startswith("S2") else "S3"
             for eid in self._target_ids]
        )
        names = target_df["norm_name"].to_numpy(dtype=str)
        addresses = (
            target_df["norm_address"].to_numpy(dtype=str)
            if "norm_address" in target_df.columns
            else np.array([""] * n, dtype=str)
        )
        self._target_address = addresses

        # 1. Token inverted index (with document frequency capping)
        if self.cfg.enable_token_blocking:
            t_tok = time.time()
            log.info("  Building token inverted index ...")
            raw_idx: Dict[str, List[int]] = defaultdict(list)
            for pos, name in enumerate(names):
                for tok in self._tokenize(name):
                    raw_idx[tok].append(pos)

            # Filter high-frequency noise tokens to bound query latency and prune false candidates
            max_freq = self.cfg.max_token_frequency
            self._token_index = {
                tok: pos_list for tok, pos_list in raw_idx.items()
                if len(pos_list) <= max_freq
            }
            n_pruned = len(raw_idx) - len(self._token_index)
            log.info("    -> %d informative tokens indexed (%d high-frequency tokens pruned) in %.2fs",
                     len(self._token_index), n_pruned, time.time() - t_tok)
            del raw_idx
            gc.collect()

        # 1b. Phonetic Soundex inverted index (Section 1.4)
        if self.cfg.enable_phonetic_blocking:
            t_ph = time.time()
            log.info("  Building phonetic Soundex inverted index ...")
            raw_ph: Dict[str, List[int]] = defaultdict(list)
            for pos, name in enumerate(names):
                toks = [tok for tok in name.split() if len(tok) >= self.cfg.min_token_length][:2]
                for tok in toks:
                    sx = soundex(tok)
                    if sx:
                        raw_ph[sx].append(pos)
            max_freq = self.cfg.max_token_frequency
            self._phonetic_index = {
                sx: pos_list for sx, pos_list in raw_ph.items()
                if len(pos_list) <= max_freq
            }
            log.info("    -> %d phonetic keys indexed in %.2fs",
                     len(self._phonetic_index), time.time() - t_ph)
            del raw_ph
            gc.collect()

        # Query neighbor count cannot exceed target population
        query_k = min(self._nn_query_k, n)

        # Adaptive min_df for tiny batches/unit tests
        eff_min_df = 1 if n < 3 else 2

        # 2. Char n-gram TF-IDF + NearestNeighbors
        if self.cfg.enable_ngram_tfidf:
            t_ng = time.time()
            log.info("  Fitting char n-gram TF-IDF (range=%s, max_feat=%d) ...",
                     self.cfg.char_ngram_range, self.cfg.tfidf_max_features)
            self._ng_vec = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=self.cfg.char_ngram_range,
                max_features=self.cfg.tfidf_max_features,
                sublinear_tf=True,
                min_df=eff_min_df,
                dtype=np.float32,
            )
            self._ng_mat = self._ng_vec.fit_transform(names)
            log.info("    -> char TF-IDF matrix: %s, nnz=%d in %.2fs",
                     self._ng_mat.shape, self._ng_mat.nnz, time.time() - t_ng)

            log.info("  Initializing NearestNeighbors for char n-gram TF-IDF (k=%d, brute, n_jobs=%d) ...",
                     query_k, self.cfg.n_jobs)
            self._ng_nn = NearestNeighbors(
                n_neighbors=query_k,
                metric="cosine",
                algorithm="brute",
                n_jobs=self.cfg.n_jobs,
            )
            self._ng_nn.fit(self._ng_mat)

        # 3. Word TF-IDF + NearestNeighbors
        if self.cfg.enable_word_tfidf:
            t_wd = time.time()
            log.info("  Fitting word TF-IDF (range=%s, max_feat=%d) ...",
                     self.cfg.word_ngram_range, self.cfg.tfidf_max_features)
            self._word_vec = TfidfVectorizer(
                analyzer="word",
                ngram_range=self.cfg.word_ngram_range,
                max_features=self.cfg.tfidf_max_features,
                sublinear_tf=True,
                min_df=eff_min_df,
                dtype=np.float32,
            )
            self._word_mat = self._word_vec.fit_transform(names)
            log.info("    -> word TF-IDF matrix: %s, nnz=%d in %.2fs",
                     self._word_mat.shape, self._word_mat.nnz, time.time() - t_wd)

            log.info("  Initializing NearestNeighbors for word TF-IDF (k=%d, brute, n_jobs=%d) ...",
                     query_k, self.cfg.n_jobs)
            self._word_nn = NearestNeighbors(
                n_neighbors=query_k,
                metric="cosine",
                algorithm="brute",
                n_jobs=self.cfg.n_jobs,
            )
            self._word_nn.fit(self._word_mat)

        # 4. Address TF-IDF + NearestNeighbors (Section 1.1)
        if self.cfg.enable_address_blocking:
            t_addr = time.time()
            log.info("  Fitting address TF-IDF (char 2-4, max_feat=%d) ...",
                     self.cfg.tfidf_max_features)
            self._addr_vec = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                max_features=self.cfg.tfidf_max_features,
                sublinear_tf=True,
                min_df=eff_min_df,
                dtype=np.float32,
            )
            self._addr_mat = self._addr_vec.fit_transform(addresses)
            query_k_addr = min(self.cfg.top_k_addr * 2, n)
            log.info("    -> address TF-IDF matrix: %s, nnz=%d in %.2fs",
                     self._addr_mat.shape, self._addr_mat.nnz, time.time() - t_addr)
            self._addr_nn = NearestNeighbors(
                n_neighbors=query_k_addr,
                metric="cosine",
                algorithm="brute",
                n_jobs=self.cfg.n_jobs,
            )
            self._addr_nn.fit(self._addr_mat)

        log.info("Index building complete in %.1fs | Mem: %.1f MB",
                 time.time() - t0, get_mem_mb())

    # ── candidate generation ──────────────────────────────────────────────

    def generate_candidates(self, s1_df: pd.DataFrame) -> pd.DataFrame:
        """
        Query all indices for every S1 entity in configurable batches and return
        a deduplicated candidate DataFrame across unioned channels.
        """
        if self._target_ids is None:
            raise RuntimeError("Call build_indices() before generate_candidates().")

        cfg = self.cfg
        t_start = time.time()
        peak_mem = get_mem_mb()

        # Storage: {(s1_id, cand_id): set of reasons}
        pair_reasons: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

        s1_ids       = s1_df["entity_id"].to_numpy(dtype=str)
        s1_names     = s1_df["norm_name"].to_numpy(dtype=str)
        s1_countries = s1_df["norm_country"].to_numpy(dtype=str)
        s1_addrs     = (
            s1_df["norm_address"].to_numpy(dtype=str)
            if "norm_address" in s1_df.columns
            else np.array([""] * len(s1_df), dtype=str)
        )
        n_total      = len(s1_df)

        batch_size = cfg.batch_size
        n_batches  = (n_total + batch_size - 1) // batch_size
        log.info("Generating candidates for %d S1 entities in %d batches (batch_size=%d) ...",
                 n_total, n_batches, batch_size)

        for b_idx in range(n_batches):
            b_t0 = time.time()
            start = b_idx * batch_size
            end   = min(start + batch_size, n_total)

            b_ids       = s1_ids[start:end]
            b_names     = s1_names[start:end]
            b_countries = s1_countries[start:end]
            b_addrs     = s1_addrs[start:end]
            b_size      = len(b_ids)

            # ── 1. Token blocking ─────────────────────────────────────────
            if cfg.enable_token_blocking and self._token_index:
                for s1_id, name, s1_ctry in zip(b_ids, b_names, b_countries):
                    cand_counts: Dict[int, int] = defaultdict(int)
                    for tok in self._tokenize(name):
                        for pos in self._token_index.get(tok, []):
                            cand_counts[pos] += 1

                    added = 0
                    for pos, _ in sorted(cand_counts.items(), key=lambda x: -x[1]):
                        if added >= cfg.max_candidates_per_s1:
                            break
                        if self._country_ok(s1_ctry, self._target_country[pos]):
                            pair_reasons[(s1_id, self._target_ids[pos])].add(REASON_TOKEN)
                            added += 1

            # ── 1b. Phonetic blocking (Section 1.4) ───────────────────────
            if cfg.enable_phonetic_blocking and self._phonetic_index:
                for s1_id, name, s1_ctry in zip(b_ids, b_names, b_countries):
                    cand_counts_ph: Dict[int, int] = defaultdict(int)
                    toks = [tok for tok in name.split() if len(tok) >= cfg.min_token_length][:2]
                    for tok in toks:
                        sx = soundex(tok)
                        if sx and sx in self._phonetic_index:
                            for pos in self._phonetic_index[sx]:
                                cand_counts_ph[pos] += 1

                    added = 0
                    for pos, _ in sorted(cand_counts_ph.items(), key=lambda x: -x[1]):
                        if added >= (cfg.max_candidates_per_s1 // 2):
                            break
                        if self._country_ok(s1_ctry, self._target_country[pos]):
                            pair_reasons[(s1_id, self._target_ids[pos])].add(REASON_PHONETIC)
                            added += 1

            # ── 2. Char n-gram TF-IDF via NearestNeighbors ────────────────
            if cfg.enable_ngram_tfidf and self._ng_nn is not None and self._ng_vec is not None:
                q_ng = self._ng_vec.transform(b_names)
                dists_ng, indices_ng = self._ng_nn.kneighbors(q_ng)

                for i in range(b_size):
                    s1_id   = b_ids[i]
                    s1_ctry = b_countries[i]
                    added   = 0
                    for dist, pos in zip(dists_ng[i], indices_ng[i]):
                        if dist >= 1.0:
                            continue
                        sim = 1.0 - dist
                        country_match = self._country_ok(s1_ctry, self._target_country[pos])

                        if country_match:
                            pair_reasons[(s1_id, self._target_ids[pos])].add(REASON_NGRAM)
                            added += 1
                        elif cfg.enable_safety_net and sim >= cfg.safety_net_similarity_floor:
                            # Section 1.2: Safety-net pass recovering high similarity despite dirty country
                            pair_reasons[(s1_id, self._target_ids[pos])].add(REASON_SAFETY_NET)
                            added += 1

                        if added >= cfg.top_k:
                            break

            # ── 3. Word TF-IDF via NearestNeighbors ───────────────────────
            if cfg.enable_word_tfidf and self._word_nn is not None and self._word_vec is not None:
                q_word = self._word_vec.transform(b_names)
                dists_word, indices_word = self._word_nn.kneighbors(q_word)

                for i in range(b_size):
                    s1_id   = b_ids[i]
                    s1_ctry = b_countries[i]
                    added   = 0
                    for dist, pos in zip(dists_word[i], indices_word[i]):
                        if dist >= 1.0:
                            continue
                        sim = 1.0 - dist
                        country_match = self._country_ok(s1_ctry, self._target_country[pos])

                        if country_match:
                            pair_reasons[(s1_id, self._target_ids[pos])].add(REASON_WORD)
                            added += 1
                        elif cfg.enable_safety_net and sim >= cfg.safety_net_similarity_floor:
                            pair_reasons[(s1_id, self._target_ids[pos])].add(REASON_SAFETY_NET)
                            added += 1

                        if added >= cfg.top_k:
                            break

            # ── 4. Address TF-IDF via NearestNeighbors (Section 1.1) ──────
            if cfg.enable_address_blocking and self._addr_nn is not None and self._addr_vec is not None:
                # Only query entities with non-empty addresses
                has_addr_mask = [len(a.strip()) > 3 for a in b_addrs]
                if any(has_addr_mask):
                    q_addr = self._addr_vec.transform(b_addrs)
                    dists_addr, indices_addr = self._addr_nn.kneighbors(q_addr)

                    for i in range(b_size):
                        if not has_addr_mask[i]:
                            continue
                        s1_id   = b_ids[i]
                        s1_ctry = b_countries[i]
                        added   = 0
                        for dist, pos in zip(dists_addr[i], indices_addr[i]):
                            if dist >= 1.0:
                                continue
                            sim = 1.0 - dist
                            country_match = self._country_ok(s1_ctry, self._target_country[pos])

                            if country_match:
                                pair_reasons[(s1_id, self._target_ids[pos])].add(REASON_ADDR_TFIDF)
                                added += 1
                            elif cfg.enable_safety_net and sim >= cfg.safety_net_similarity_floor:
                                pair_reasons[(s1_id, self._target_ids[pos])].add(REASON_SAFETY_NET)
                                added += 1

                            if added >= cfg.top_k_addr:
                                break

            # ── Progress reporting after each batch ───────────────────────
            curr_mem = get_mem_mb()
            peak_mem = max(peak_mem, curr_mem)
            elapsed  = time.time() - t_start
            b_dur    = time.time() - b_t0

            log.info(
                "Batch %d/%d (%.1f%%) | Total Candidates: %d | Batch: %.2fs | Elapsed: %.1fs | Mem: %.1f MB (Peak: %.1f MB)",
                b_idx + 1,
                n_batches,
                100.0 * (b_idx + 1) / n_batches,
                len(pair_reasons),
                b_dur,
                elapsed,
                curr_mem,
                peak_mem,
            )

        log.info("Candidate generation finished in %.1fs | Unique pairs: %d | Peak Mem: %.1f MB",
                 time.time() - t_start, len(pair_reasons), peak_mem)

        # ── Build output DataFrame ─────────────────────────────────────────
        if not pair_reasons:
            return pd.DataFrame(columns=CANDIDATE_COLS)

        id_to_pos = {eid: i for i, eid in enumerate(self._target_ids)}

        records = []
        for (s1_id, cand_id), reasons in pair_reasons.items():
            pos = id_to_pos.get(cand_id)
            if pos is None:
                continue
            records.append({
                "source1_entity_id":  s1_id,
                "candidate_entity_id": cand_id,
                "source_dataset":     self._target_src[pos],
                "blocking_reason":    "|".join(sorted(reasons)),
            })

        result = pd.DataFrame(records, columns=CANDIDATE_COLS)
        log.info("Final candidate DataFrame: %d rows", len(result))
        return result

    # ── evaluation & recall audit ─────────────────────────────────────────

    @staticmethod
    def evaluate(
        candidates_df: pd.DataFrame,
        ground_truth: Dict[str, Set[str]],
        s1_total: int,
        target_total: int,
        runtime_sec: float = 0.0,
        peak_mem_mb: float = 0.0,
        expected_s1_ids: Optional[Set[str]] = None,
    ) -> Dict[str, float]:
        """Alias for audit_blocking_recall for backward compatibility."""
        return Blocking.audit_blocking_recall(
            candidates_df=candidates_df,
            ground_truth=ground_truth,
            s1_total=s1_total,
            target_total=target_total,
            runtime_sec=runtime_sec,
            peak_mem_mb=peak_mem_mb,
            expected_s1_ids=expected_s1_ids,
        )

    @staticmethod
    def audit_blocking_recall(
        candidates_df: pd.DataFrame,
        ground_truth: Dict[str, Set[str]],
        s1_total: int,
        target_total: int,
        runtime_sec: float = 0.0,
        peak_mem_mb: float = 0.0,
        expected_s1_ids: Optional[Set[str]] = None,
    ) -> Dict[str, float]:
        """
        Mandatory blocking-recall audit (Section 1.6).
        Computes recall ceiling per source pair:
          - overall recall ceiling against all expected S1 entities
          - S1 <-> S2 recall ceiling
          - S1 <-> S3 recall ceiling
          - reduction ratio
        """
        total_possible = s1_total * target_total
        n_cands        = len(candidates_df)

        pair_set: Set[Tuple[str, str]] = set(
            zip(candidates_df["source1_entity_id"],
                candidates_df["candidate_entity_id"])
        )

        s1_covered = candidates_df["source1_entity_id"].nunique()
        s1_present = set(candidates_df["source1_entity_id"].unique())
        eval_s1_set = expected_s1_ids if expected_s1_ids is not None else s1_present
        gt_filtered = {k: v for k, v in ground_truth.items() if k in eval_s1_set}

        true_total = sum(len(v) for v in gt_filtered.values())
        found_total = 0
        s2_true_total = 0
        s2_found = 0
        s3_true_total = 0
        s3_found = 0

        for s1_id, matched in gt_filtered.items():
            for cand_id in matched:
                is_found = (s1_id, cand_id) in pair_set
                if is_found:
                    found_total += 1
                if cand_id.startswith("S2"):
                    s2_true_total += 1
                    if is_found:
                        s2_found += 1
                elif cand_id.startswith("S3"):
                    s3_true_total += 1
                    if is_found:
                        s3_found += 1

        recall    = found_total / true_total if true_total > 0 else 0.0
        recall_s2 = s2_found / s2_true_total if s2_true_total > 0 else 0.0
        recall_s3 = s3_found / s3_true_total if s3_true_total > 0 else 0.0
        reduction_ratio = 1.0 - n_cands / total_possible if total_possible > 0 else 0.0
        avg_per_s1 = n_cands / s1_covered if s1_covered > 0 else 0.0

        audit = {
            "candidate_pairs":             n_cands,
            "reduction_ratio":             reduction_ratio,
            "candidate_recall":            recall,
            "candidate_recall_overall":    recall,
            "candidate_recall_s1_s2":      recall_s2,
            "candidate_recall_s1_s3":      recall_s3,
            "found_true_pairs":            found_total,
            "true_pairs_total":            true_total,
            "found_s2_pairs":              s2_found,
            "true_s2_total":               s2_true_total,
            "found_s3_pairs":              s3_found,
            "true_s3_total":               s3_true_total,
            "avg_candidates_per_s1":       avg_per_s1,
            "s1_entities_with_candidates": s1_covered,
            "total_runtime_sec":           runtime_sec,
            "peak_memory_mb":              peak_mem_mb,
        }
        log.info("Blocking Recall Audit (Section 1.6):")
        log.info("  Overall Recall Ceiling : %.2f%% (%d / %d)", recall * 100, found_total, true_total)
        log.info("  S1 <-> S2 Recall Ceiling: %.2f%% (%d / %d)", recall_s2 * 100, s2_found, s2_true_total)
        log.info("  S1 <-> S3 Recall Ceiling: %.2f%% (%d / %d)", recall_s3 * 100, s3_found, s3_true_total)
        log.info("  Reduction Ratio         : %.6f (1 - %d / %d)", reduction_ratio, n_cands, total_possible)
        return audit

    # ── I/O ───────────────────────────────────────────────────────────────

    @staticmethod
    def save_candidates(candidates_df: pd.DataFrame, path: str) -> None:
        """
        Write candidate pairs to a tab-separated file.

        Parameters
        ----------
        candidates_df : pd.DataFrame
        path : str
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        candidates_df.to_csv(path, sep="\t", index=False)
        log.info("Saved %d candidate pairs to %s", len(candidates_df), path)

    # ── private helpers ───────────────────────────────────────────────────

    def _tokenize(self, name: str) -> List[str]:
        """
        Split a normalised name into significant unigrams.
        Tokens shorter than ``cfg.min_token_length`` are discarded.
        """
        return [tok for tok in name.split()
                if len(tok) >= self.cfg.min_token_length]

    def _country_ok(self, s1_country: str, cand_country: str) -> bool:
        """
        True if the two country strings are compatible for blocking.

        Rules:
        - If country gate is disabled, always True.
        - If either side is empty (missing country), always True.
        - Otherwise, must be identical after normalisation.
        """
        if not self.cfg.enable_country_gate:
            return True
        if not s1_country or not cand_country:
            return True
        return s1_country == cand_country


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

def run_blocking(
    cfg: Optional[BlockingConfig] = None,
    nrows: Optional[int] = None,
    save_output: bool = True,
) -> Tuple[pd.DataFrame, Optional[Dict[str, float]]]:
    """
    Full blocking pipeline:
        load data -> build indices -> generate candidates
        -> evaluate (if ground truth available) -> save TSV.
    """
    paths   = PathConfig()
    cfg     = cfg or BlockingConfig()
    t_start = time.time()

    # 1. Load data
    s1_df = load_source_df(str(paths.train_dir / paths.train_s1_file), nrows=nrows)
    s1_ids_set = set(s1_df["entity_id"].tolist())
    gt_path = paths.train_dir / paths.train_ground_truth_file

    if nrows is not None and gt_path.exists():
        log.info("Sampling %d S1 entities: Preserving all true target alignments in target corpus ...", len(s1_df))
        gt = load_ground_truth(str(gt_path))
        needed_targets = set()
        for s1 in s1_ids_set:
            needed_targets.update(gt.get(s1, set()))

        s2_needed = {m for m in needed_targets if m.startswith("S2")}
        s3_needed = {m for m in needed_targets if m.startswith("S3")}

        # Load S2 preserving all true targets + distractor records
        s2_raw = load_source_df(str(paths.train_dir / paths.train_s2_file))
        s2_df = s2_raw[s2_raw["entity_id"].isin(s2_needed) | (s2_raw.index < max(nrows, 5000))].copy()
        del s2_raw

        # Load S3 preserving all true targets + distractor records
        s3_raw = load_source_df(str(paths.train_dir / paths.train_s3_file))
        s3_df = s3_raw[s3_raw["entity_id"].isin(s3_needed) | (s3_raw.index < max(nrows, 5000))].copy()
        del s3_raw
        gc.collect()
    else:
        s2_df = load_source_df(str(paths.train_dir / paths.train_s2_file), nrows=nrows)
        s3_df = load_source_df(str(paths.train_dir / paths.train_s3_file), nrows=nrows)

    # 2. Build combined target (S2 + S3)
    log.info("Concatenating S2 (%d) + S3 (%d) into target corpus ...",
             len(s2_df), len(s3_df))
    target_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    gc.collect()

    # 3. Build indices
    blocker = Blocking(cfg)
    blocker.build_indices(target_df)

    # 4. Generate candidates
    candidates_df = blocker.generate_candidates(s1_df)
    total_time    = time.time() - t_start
    peak_mem      = get_mem_mb()

    # 5. Evaluate
    metrics = None
    if gt_path.exists():
        gt = load_ground_truth(str(gt_path))
        metrics = Blocking.evaluate(
            candidates_df,
            ground_truth = gt,
            s1_total     = len(s1_df),
            target_total = len(target_df),
            runtime_sec  = total_time,
            peak_mem_mb  = peak_mem,
            expected_s1_ids = s1_ids_set,
        )
        log.info("════════════════ Blocking Evaluation Summary ════════════════")
        log.info("  Candidate Pairs               : %d",   metrics["candidate_pairs"])
        log.info("  Reduction Ratio               : %.6f", metrics["reduction_ratio"])
        log.info("  Candidate Recall              : %.4f", metrics["candidate_recall"])
        log.info("  Found / Total True Pairs      : %d / %d",
                 metrics["found_true_pairs"], metrics["true_pairs_total"])
        log.info("  Avg Candidates per S1 Entity  : %.2f", metrics["avg_candidates_per_s1"])
        log.info("  S1 Entities with Candidates   : %d",   metrics["s1_entities_with_candidates"])
        log.info("  Peak Memory Usage             : %.1f MB", metrics["peak_memory_mb"])
        log.info("  Total Runtime                 : %.1f s", metrics["total_runtime_sec"])
        log.info("═════════════════════════════════════════════════════════════")

    # 6. Save
    if save_output:
        out_path = paths.output_dir / paths.candidate_pairs_file
        Blocking.save_candidates(candidates_df, str(out_path))

    return candidates_df, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Demo / smoke-test
# ─────────────────────────────────────────────────────────────────────────────

def _run_demo() -> None:
    """
    Test section:
    Loads sample records, runs each blocking method separately and hybrid,
    and prints evaluation metrics.
    """
    SAMPLE = 5_000
    SEP = "=" * 68

    paths = PathConfig()
    gt_all = load_ground_truth(str(paths.train_dir / paths.train_ground_truth_file))

    log.info("Loading %d sample rows from each source for testing ...", SAMPLE)
    s1_df = load_source_df(str(paths.train_dir / paths.train_s1_file), nrows=SAMPLE)
    s2_df = load_source_df(str(paths.train_dir / paths.train_s2_file), nrows=SAMPLE)
    s3_df = load_source_df(str(paths.train_dir / paths.train_s3_file), nrows=SAMPLE)
    target_sample = pd.concat([s2_df, s3_df], ignore_index=True)

    s1_ids_set = set(s1_df["entity_id"].tolist())
    gt_sample  = {k: v for k, v in gt_all.items() if k in s1_ids_set}
    total_gt_pairs = sum(len(v) for v in gt_sample.values())
    log.info("Sample S1 entities: %d with %d true pairs in full ground truth",
             len(gt_sample), total_gt_pairs)

    def _eval_print(label: str, cands: pd.DataFrame, t_dur: float, p_mem: float) -> None:
        m = Blocking.evaluate(
            cands,
            ground_truth = gt_sample,
            s1_total     = len(s1_df),
            target_total = len(target_sample),
            runtime_sec  = t_dur,
            peak_mem_mb  = p_mem,
        )
        print(f"\n{SEP}")
        print(f"  {label}")
        print(SEP)
        print(f"  Candidate pairs        : {m['candidate_pairs']:>10,}")
        print(f"  Reduction ratio        : {m['reduction_ratio']:>10.6f}")
        print(f"  Candidate recall       : {m['candidate_recall']:>10.4f}  (sample test)")
        print(f"  Found / true pairs     : {m['found_true_pairs']:>7,} / {m['true_pairs_total']:,}")
        print(f"  Avg candidates / S1    : {m['avg_candidates_per_s1']:>10.1f}")
        print(f"  Peak memory            : {m['peak_memory_mb']:>10.1f} MB")
        print(f"  Runtime                : {m['total_runtime_sec']:>10.2f} s")

    # ── 1. Token Blocking Only ────────────────────────────────────────────
    cfg_tok = BlockingConfig(
        enable_token_blocking = True,
        enable_ngram_tfidf    = False,
        enable_word_tfidf     = False,
        batch_size            = 2_500,
    )
    t0 = time.time()
    b_tok = Blocking(cfg_tok)
    b_tok.build_indices(target_sample)
    cands_tok = b_tok.generate_candidates(s1_df)
    _eval_print("1. Token Blocking Only", cands_tok, time.time() - t0, get_mem_mb())

    # ── 2. Char N-gram TF-IDF Only ────────────────────────────────────────
    cfg_ng = BlockingConfig(
        enable_token_blocking = False,
        enable_ngram_tfidf    = True,
        enable_word_tfidf     = False,
        top_k                 = 30,
        batch_size            = 2_500,
    )
    t0 = time.time()
    b_ng = Blocking(cfg_ng)
    b_ng.build_indices(target_sample)
    cands_ng = b_ng.generate_candidates(s1_df)
    _eval_print("2. Char N-gram TF-IDF Only (NearestNeighbors)", cands_ng, time.time() - t0, get_mem_mb())

    # ── 3. Word TF-IDF Only ───────────────────────────────────────────────
    cfg_wd = BlockingConfig(
        enable_token_blocking = False,
        enable_ngram_tfidf    = False,
        enable_word_tfidf     = True,
        top_k                 = 30,
        batch_size            = 2_500,
    )
    t0 = time.time()
    b_wd = Blocking(cfg_wd)
    b_wd.build_indices(target_sample)
    cands_wd = b_wd.generate_candidates(s1_df)
    _eval_print("3. Word TF-IDF Only (NearestNeighbors)", cands_wd, time.time() - t0, get_mem_mb())

    # ── 4. Hybrid Blocking ────────────────────────────────────────────────
    cfg_hybrid = BlockingConfig(
        enable_token_blocking = True,
        enable_ngram_tfidf    = True,
        enable_word_tfidf     = True,
        top_k                 = 30,
        max_candidates_per_s1 = 100,
        batch_size            = 2_500,
    )
    t0 = time.time()
    b_hyb = Blocking(cfg_hybrid)
    b_hyb.build_indices(target_sample)
    cands_hyb = b_hyb.generate_candidates(s1_df)
    _eval_print("4. Hybrid Blocking (Token + Char TF-IDF + Word TF-IDF)", cands_hyb, time.time() - t0, get_mem_mb())

    print(f"\n{SEP}")
    print("  Blocking_reason distribution (Hybrid):")
    print(SEP)
    reason_counts = (
        cands_hyb["blocking_reason"]
        .str.split("|")
        .explode()
        .value_counts()
    )
    for reason, cnt in reason_counts.items():
        if reason:
            print(f"    {reason:35s}: {cnt:>8,}")

    print(f"\n{SEP}")
    print("Test complete. The pipeline is memory-bounded and scalable.")
    print(SEP)


if __name__ == "__main__":
    _run_demo()
