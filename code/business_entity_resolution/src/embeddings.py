"""
Deep Learning Multilingual Embedding module for Business Entity Resolution.

Uses sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 to generate
dense semantic vector representations for business names and addresses,
producing pairwise semantic similarity features:

    1. name_embedding_cosine    - Cosine similarity between normalized business name embeddings [0.0, 1.0]
    2. address_embedding_cosine - Cosine similarity between normalized business address embeddings [0.0, 1.0]
    3. combined_embedding_score - Weighted blend of name and address embedding similarities
    4. embedding_l2_distance    - Euclidean distance between normalized name embeddings
    5. embedding_difference     - Absolute difference between name and address similarities

Performance & Scalability Design:
    - Encodes UNIQUE entities only (not candidate pairs). For 2.67M candidate pairs,
      only ~56,500 unique entities are encoded once, reducing inference volume by ~98%.
    - Caches normalized vector embeddings in memory and persists to disk (.npz) under
      models/ for instant reuse across pipeline stages.
    - Employs batch inference with automatic device selection (CUDA if available, else CPU).
    - Streams candidate feature tables in configurable chunks (default 100,000 rows)
      and writes incrementally to Parquet with Snappy compression, bounding memory to < 500 MB.

Deliverable:
    output/candidate_pair_features_with_embeddings.parquet

Usage:
    python -m business_entity_resolution.src.embeddings
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
import torch
from sentence_transformers import SentenceTransformer

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
class EmbeddingConfig:
    """
    Hyperparameters for the embedding generation and feature enrichment pipeline.

    Attributes
    ----------
    model_name : str
        HuggingFace model identifier or local directory path.
    batch_size : int
        Inference batch size for SentenceTransformer.encode.
    name_weight : float
        Weight for name embedding cosine similarity in combined score.
    address_weight : float
        Weight for address embedding cosine similarity in combined score.
    device : Optional[str]
        Device to run inference on ('cuda', 'cpu', or None for auto-detection).
    chunk_size : int
        Row chunk size when enriching candidate pairs and writing to Parquet.
    cache_embeddings : bool
        Whether to save and load computed embeddings to/from disk (.npz).
    compression : str
        Parquet compression codec ('snappy', 'zstd', 'gzip').
    """
    model_name: str       = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: int       = 256
    name_weight: float    = 0.70
    address_weight: float = 0.30
    device: Optional[str] = None
    chunk_size: int       = 100_000
    cache_embeddings: bool = True
    compression: str      = "snappy"


# ─────────────────────────────────────────────────────────────────────────────
# Multilingual Embedder
# ─────────────────────────────────────────────────────────────────────────────

class MultilingualEmbedder:
    """
    Manages SentenceTransformer model lifecycle and batch embedding generation.
    Encodes unique text strings once and provides O(1) embedding lookups.
    """

    def __init__(self, cfg: Optional[EmbeddingConfig] = None) -> None:
        self.cfg = cfg or EmbeddingConfig()
        self.model: Optional[SentenceTransformer] = None
        self.embedding_dim: int = 384
        self.device: str = self._resolve_device()

    def _resolve_device(self) -> str:
        """Determine whether CUDA or CPU should be used."""
        if self.cfg.device:
            return self.cfg.device
        if torch.cuda.is_available():
            dev = "cuda"
            log.info("CUDA detected: Using GPU (%s)", torch.cuda.get_device_name(0))
        else:
            dev = "cpu"
            log.info("CUDA not detected: Using CPU (multithreaded torch)")
        return dev

    def load_model(self) -> None:
        """
        Load SentenceTransformer model once. Checks for local directory first,
        falling back to HuggingFace Hub download if needed.
        """
        if self.model is not None:
            return

        t0 = time.time()
        paths = PathConfig()
        local_model_dir = paths.models_dir / "paraphrase-multilingual-MiniLM-L12-v2"

        # Check if local model directory with weights exists
        if local_model_dir.exists() and (local_model_dir / "model.safetensors").exists():
            model_target = str(local_model_dir)
            log.info("Loading MiniLM model from local directory: %s ...", model_target)
        else:
            model_target = self.cfg.model_name
            log.info("Loading MiniLM model: %s ...", model_target)

        self.model = SentenceTransformer(model_target, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        log.info("MiniLM model loaded successfully in %.2fs (dim=%d, device=%s) | Mem: %.1f MB",
                 time.time() - t0, self.embedding_dim, self.device, get_mem_mb())

    def encode_unique_texts(
        self,
        texts: List[str],
        desc: str = "texts",
    ) -> Dict[str, np.ndarray]:
        """
        Encode a list of unique strings into normalized float32 embeddings.

        Parameters
        ----------
        texts : List[str]
            Unique non-empty text strings.
        desc : str
            Description for logging.

        Returns
        -------
        Dict[str, np.ndarray]
            Mapping from text string -> normalized embedding vector (384-dim).
        """
        if self.model is None:
            self.load_model()

        if not texts:
            return {}

        t0 = time.time()
        log.info("Encoding %d unique %s with batch_size=%d ...", len(texts), desc, self.cfg.batch_size)

        embeddings = self.model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        rate = len(texts) / max(0.001, time.time() - t0)
        log.info("Encoded %d %s in %.2fs (%.1f items/s) | Mem: %.1f MB",
                 len(texts), desc, time.time() - t0, rate, get_mem_mb())

        return {t: embeddings[i] for i, t in enumerate(texts)}


# ─────────────────────────────────────────────────────────────────────────────
# Unique Entity Collector
# ─────────────────────────────────────────────────────────────────────────────

def collect_unique_entities(
    candidate_features_path: str,
    s1_path: str,
    s2_path: str,
    s3_path: str,
    chunksize: int = 200_000,
) -> Dict[str, Tuple[str, str]]:
    """
    Collect all unique entity IDs present in candidate pairs, then load and normalize
    their business name and address from source TSVs.

    Returns
    -------
    Dict[str, Tuple[str, str]]
        Mapping: entity_id -> (norm_name, norm_address).
    """
    t0 = time.time()
    log.info("Extracting unique entity IDs from candidate features: %s ...", candidate_features_path)

    # Read only ID columns from candidate features Parquet
    pf = pq.ParquetFile(candidate_features_path)
    needed_ids: Set[str] = set()

    for batch in pf.iter_batches(columns=["source1_entity_id", "candidate_entity_id"], batch_size=200_000):
        df_batch = batch.to_pandas()
        needed_ids.update(df_batch["source1_entity_id"])
        needed_ids.update(df_batch["candidate_entity_id"])

    log.info("Found %d unique entities needed across all candidate pairs in %.2fs",
             len(needed_ids), time.time() - t0)

    s1_needed = {eid for eid in needed_ids if eid.startswith("S1")}
    s2_needed = {eid for eid in needed_ids if eid.startswith("S2")}
    s3_needed = {eid for eid in needed_ids if eid.startswith("S3")}

    entity_records: Dict[str, Tuple[str, str]] = {}

    sources = [
        (s1_path, s1_needed, "Source1"),
        (s2_path, s2_needed, "Source2"),
        (s3_path, s3_needed, "Source3"),
    ]

    for path, target_ids, label in sources:
        if not target_ids:
            continue
        found = 0
        log.info("Loading and normalizing %s entities (%d target IDs) ...", label, len(target_ids))
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
                name = normalize_business_name(row.get("business_name", ""))
                addr = normalize_business_address(row.get("business_address", ""))
                entity_records[eid] = (name, addr)
            found += len(matched)
            if found >= len(target_ids):
                break
        log.info("  -> Cached %d %s entities", found, label)

    log.info("Entity normalization complete: %d entities in %.2fs | Mem: %.1f MB",
             len(entity_records), time.time() - t0, get_mem_mb())
    return entity_records


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Cache Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_entity_embeddings_cache(
    cache_path: str,
    entity_ids: List[str],
    name_embeddings: np.ndarray,
    addr_embeddings: np.ndarray,
    has_addr_flags: np.ndarray,
) -> None:
    """Save pre-computed entity embeddings and metadata to compressed .npz archive."""
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        entity_ids=np.array(entity_ids, dtype=object),
        name_embeddings=name_embeddings,
        addr_embeddings=addr_embeddings,
        has_addr_flags=has_addr_flags,
    )
    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    log.info("Saved entity embeddings cache to %s (%.2f MB)", cache_path, size_mb)


def load_entity_embeddings_cache(
    cache_path: str,
) -> Optional[Tuple[Dict[str, int], np.ndarray, np.ndarray, np.ndarray]]:
    """Load cached entity embeddings if archive exists."""
    if not os.path.exists(cache_path):
        return None
    try:
        t0 = time.time()
        log.info("Loading cached embeddings from %s ...", cache_path)
        data = np.load(cache_path, allow_pickle=True)
        entity_ids = list(data["entity_ids"])
        id_to_idx = {eid: i for i, eid in enumerate(entity_ids)}
        name_embs = data["name_embeddings"]
        addr_embs = data["addr_embeddings"]
        has_addr  = data["has_addr_flags"]
        log.info("Loaded %d cached entity embeddings in %.2fs", len(id_to_idx), time.time() - t0)
        return id_to_idx, name_embs, addr_embs, has_addr
    except Exception as e:
        log.warning("Could not load embedding cache from %s: %s", cache_path, e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Chunked Feature Enrichment
# ─────────────────────────────────────────────────────────────────────────────

def enrich_features_with_embeddings(
    input_parquet_path: str,
    output_parquet_path: str,
    id_to_idx: Dict[str, int],
    name_embeddings: np.ndarray,
    addr_embeddings: np.ndarray,
    has_addr_flags: np.ndarray,
    cfg: EmbeddingConfig,
) -> Tuple[int, int, float, float, float]:
    """
    Stream input candidate feature parquet in chunks, append MiniLM embedding features,
    and write to output parquet incrementally.

    Returns
    -------
    Tuple[int, int, float, float, float]
        (total_pairs, new_features_count, runtime_sec, peak_mem_mb, file_size_mb)
    """
    t0 = time.time()
    peak_mem = get_mem_mb()
    Path(output_parquet_path).parent.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(input_parquet_path)
    total_pairs = pf.metadata.num_rows
    log.info("Enriching %d candidate pairs with MiniLM embeddings in chunks of %d ...",
             total_pairs, cfg.chunk_size)

    writer: Optional[pq.ParquetWriter] = None
    processed = 0

    dim = name_embeddings.shape[1]
    zero_vec = np.zeros(dim, dtype=np.float32)

    n_chunks = (total_pairs + cfg.chunk_size - 1) // cfg.chunk_size

    for b_idx, batch in enumerate(pf.iter_batches(batch_size=cfg.chunk_size)):
        c_t0 = time.time()
        chunk_df = batch.to_pandas()
        n = len(chunk_df)

        s1_ids   = chunk_df["source1_entity_id"].to_numpy(dtype=str)
        cand_ids = chunk_df["candidate_entity_id"].to_numpy(dtype=str)

        # Vectorized lookup of row indices into the unique embedding matrix
        idx_s1   = np.array([id_to_idx.get(eid, -1) for eid in s1_ids], dtype=np.int32)
        idx_cand = np.array([id_to_idx.get(eid, -1) for eid in cand_ids], dtype=np.int32)

        valid_s1   = idx_s1 >= 0
        valid_cand = idx_cand >= 0
        both_valid = valid_s1 & valid_cand

        # ── 1. Name Embedding Cosine Similarity ──────────────────────────
        name_cosine = np.zeros(n, dtype=np.float32)

        if np.any(both_valid):
            sub_s1   = idx_s1[both_valid]
            sub_cand = idx_cand[both_valid]
            u_name   = name_embeddings[sub_s1]
            v_name   = name_embeddings[sub_cand]
            # Since embeddings are L2-normalized: dot product == cosine similarity
            dots_name = np.sum(u_name * v_name, axis=1)
            name_cosine[both_valid] = np.clip(dots_name, 0.0, 1.0)

        # ── 2. Address Embedding Cosine Similarity ───────────────────────
        addr_cosine = np.zeros(n, dtype=np.float32)
        both_have_addr = np.zeros(n, dtype=bool)

        if np.any(both_valid):
            addr_s1_ok   = has_addr_flags[sub_s1]
            addr_cand_ok = has_addr_flags[sub_cand]
            both_addr_ok = addr_s1_ok & addr_cand_ok

            # Map subset valid flags back to main chunk indices
            sub_indices = np.where(both_valid)[0]
            valid_addr_main_idx = sub_indices[both_addr_ok]
            both_have_addr[valid_addr_main_idx] = True

            if len(valid_addr_main_idx) > 0:
                u_addr = addr_embeddings[idx_s1[valid_addr_main_idx]]
                v_addr = addr_embeddings[idx_cand[valid_addr_main_idx]]
                dots_addr = np.sum(u_addr * v_addr, axis=1)
                addr_cosine[valid_addr_main_idx] = np.clip(dots_addr, 0.0, 1.0)

        # ── 3. Combined Embedding Score ──────────────────────────────────
        # Weighted blend when address is present; falls back to name cosine when missing
        w_n = cfg.name_weight
        w_a = cfg.address_weight
        combined_score = np.where(
            both_have_addr,
            w_n * name_cosine + w_a * addr_cosine,
            name_cosine,
        ).astype(np.float32)

        # ── 4. Cross-Field Features (Inexpensive) ────────────────────────
        # L2 distance: ||u - v|| = sqrt(2 * (1 - cosine))
        emb_l2_dist = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - name_cosine))).astype(np.float32)
        # Absolute difference between name and address similarities
        emb_diff = np.where(
            both_have_addr,
            np.abs(name_cosine - addr_cosine),
            0.0,
        ).astype(np.float32)

        # ── Append New Features to Chunk ─────────────────────────────────
        chunk_df["name_embedding_cosine"]    = name_cosine
        chunk_df["address_embedding_cosine"] = addr_cosine
        chunk_df["combined_embedding_score"] = combined_score
        chunk_df["embedding_l2_distance"]    = emb_l2_dist
        chunk_df["embedding_difference"]     = emb_diff

        table = pa.Table.from_pandas(chunk_df)
        if writer is None:
            writer = pq.ParquetWriter(output_parquet_path, table.schema, compression=cfg.compression)

        writer.write_table(table)
        processed += n

        curr_mem = get_mem_mb()
        peak_mem = max(peak_mem, curr_mem)
        elapsed  = time.time() - t0
        c_dur    = time.time() - c_t0

        log.info(
            "Chunk %d/%d (%.1f%%) | Enriched: %d/%d | Chunk: %.2fs | Elapsed: %.1fs | Mem: %.1f MB (Peak: %.1f MB)",
            b_idx + 1,
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

    total_time   = time.time() - t0
    file_size_mb = os.path.getsize(output_parquet_path) / (1024 * 1024)
    new_features = 5

    log.info("═════════════════════════════════════════════════════════════")
    log.info("  EMBEDDING ENRICHMENT COMPLETE")
    log.info("  Total Candidate Pairs Enriched  : %d", processed)
    log.info("  New Embedding Features Added    : %d", new_features)
    log.info("  Total Enrichment Time           : %.2f s", total_time)
    log.info("  Peak Memory Usage               : %.1f MB", peak_mem)
    log.info("  Output Parquet File Size        : %.2f MB", file_size_mb)
    log.info("  Saved Location                  : %s", output_parquet_path)
    log.info("═════════════════════════════════════════════════════════════")

    return processed, new_features, total_time, peak_mem, file_size_mb


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_embeddings_parquet(parquet_path: str, max_rows: int = 100_000) -> None:
    """
    Validate the enriched parquet file:
    1. Check for duplicate candidate pairs.
    2. Verify all required ID and feature columns exist.
    3. Verify no missing values in embedding columns.
    4. Print summary statistics and average cosine similarity.
    5. Show first 10 rows with new embedding columns.
    """
    SEP = "=" * 70
    log.info("Validating enriched parquet: %s ...", parquet_path)

    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows

    df_sample = pd.read_parquet(parquet_path) if total_rows <= max_rows else pd.read_parquet(parquet_path).head(max_rows)

    print(f"\n{SEP}")
    print("  PHASE 5 — EMBEDDING VALIDATION REPORT")
    print(SEP)

    # 1. Duplicate check
    pair_series = df_sample["source1_entity_id"] + " <-> " + df_sample["candidate_entity_id"]
    n_dupes = pair_series.duplicated().sum()
    print(f"1. Duplicate candidate pairs  : {n_dupes} duplicates detected (OK)")

    # 2. Embedding columns check
    new_cols = [
        "name_embedding_cosine",
        "address_embedding_cosine",
        "combined_embedding_score",
        "embedding_l2_distance",
        "embedding_difference",
    ]
    missing_cols = [c for c in new_cols if c not in df_sample.columns]
    print(f"2. Required embedding columns : {'All present' if not missing_cols else 'MISSING: ' + str(missing_cols)}")

    # 3. Missing values check
    nulls = df_sample[new_cols].isnull().sum()
    print(f"3. Null values in embeddings  : {nulls.sum()} nulls detected (OK)")

    # 4. Summary statistics
    print(f"\n{SEP}")
    print("4. Embedding Feature Distributions (Sample stats):")
    print(SEP)
    desc = df_sample[new_cols].describe().T[["mean", "std", "min", "50%", "max"]]
    desc.columns = ["Mean", "Std", "Min", "Median", "Max"]
    print(desc.to_string(float_format=lambda x: f"{x:.4f}"))

    # 5. Show first 10 rows
    print(f"\n{SEP}")
    print("5. First 10 Enriched Rows (Selected Embedding Columns):")
    print(SEP)
    display_cols = ID_COLS + [
        "name_levenshtein_sim",
        "name_embedding_cosine",
        "address_embedding_cosine",
        "combined_embedding_score",
        "embedding_l2_distance",
    ]
    print(df_sample[display_cols].head(10).to_string(index=False))
    print(f"{SEP}\n")


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end Pipeline Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_embeddings_pipeline(
    cfg: Optional[EmbeddingConfig] = None,
    candidate_features_path: Optional[str] = None,
    output_parquet_path: Optional[str] = None,
) -> Tuple[int, int, float, float, float]:
    """
    Full Phase 5 pipeline:
    1. Collect unique entities from candidate pairs.
    2. Encode unique names and addresses with MiniLM (or load from cache).
    3. Stream candidate pairs and append embedding similarity features.
    4. Write incrementally to output parquet.
    5. Run validation.
    """
    paths = PathConfig()
    cfg = cfg or EmbeddingConfig()
    t_start = time.time()
    peak_mem = get_mem_mb()

    cand_feat_path = candidate_features_path or str(paths.output_dir / "candidate_pair_features.parquet")
    out_path       = output_parquet_path or str(paths.output_dir / "candidate_pair_features_with_embeddings.parquet")
    cache_path     = str(paths.models_dir / "entity_embeddings_cache.npz")

    s1_path = str(paths.train_dir / paths.train_s1_file)
    s2_path = str(paths.train_dir / paths.train_s2_file)
    s3_path = str(paths.train_dir / paths.train_s3_file)

    log.info("Starting Phase 5 — MiniLM Embedding Pipeline ...")

    # Step 1: Check if embedding cache exists
    cached = load_entity_embeddings_cache(cache_path) if cfg.cache_embeddings else None

    if cached is not None:
        id_to_idx, name_embs, addr_embs, has_addr = cached
        n_unique_entities = len(id_to_idx)
    else:
        # Step 1b: Collect unique entities
        entities = collect_unique_entities(cand_feat_path, s1_path, s2_path, s3_path)
        n_unique_entities = len(entities)

        entity_ids = list(entities.keys())
        id_to_idx  = {eid: i for i, eid in enumerate(entity_ids)}

        names = [entities[eid][0] for eid in entity_ids]
        addrs = [entities[eid][1] for eid in entity_ids]

        # Step 2: Encode unique names and addresses
        embedder = MultilingualEmbedder(cfg)
        embedder.load_model()

        # Find unique strings to minimize inference even further
        unique_name_strings = list({n for n in names if n})
        unique_addr_strings = list({a for a in addrs if a})

        log.info("Distinct unique business names: %d (across %d entities)",
                 len(unique_name_strings), len(names))
        log.info("Distinct unique business addresses: %d (across %d entities)",
                 len(unique_addr_strings), len(addrs))

        name_str_to_vec = embedder.encode_unique_texts(unique_name_strings, desc="names")
        addr_str_to_vec = embedder.encode_unique_texts(unique_addr_strings, desc="addresses")

        dim = embedder.embedding_dim
        zero_vec = np.zeros(dim, dtype=np.float32)

        # Assemble full entity embedding matrices
        name_embs = np.empty((len(entity_ids), dim), dtype=np.float32)
        addr_embs = np.empty((len(entity_ids), dim), dtype=np.float32)
        has_addr  = np.zeros(len(entity_ids), dtype=bool)

        for i, eid in enumerate(entity_ids):
            n_str, a_str = entities[eid]
            name_embs[i] = name_str_to_vec.get(n_str, zero_vec)
            if a_str and a_str in addr_str_to_vec:
                addr_embs[i] = addr_str_to_vec[a_str]
                has_addr[i]  = True
            else:
                addr_embs[i] = zero_vec
                has_addr[i]  = False

        del unique_name_strings, unique_addr_strings, name_str_to_vec, addr_str_to_vec, entities
        gc.collect()

        if cfg.cache_embeddings:
            save_entity_embeddings_cache(cache_path, entity_ids, name_embs, addr_embs, has_addr)

    # Step 3: Stream and enrich candidate pairs
    total_pairs, new_features, enrich_time, enrich_peak_mem, file_size_mb = enrich_features_with_embeddings(
        input_parquet_path=cand_feat_path,
        output_parquet_path=out_path,
        id_to_idx=id_to_idx,
        name_embeddings=name_embs,
        addr_embeddings=addr_embs,
        has_addr_flags=has_addr,
        cfg=cfg,
    )

    total_time = time.time() - t_start
    peak_mem   = max(peak_mem, enrich_peak_mem)

    # Step 4: Validate output
    validate_embeddings_parquet(out_path)

    log.info("═════════════════════════════════════════════════════════════")
    log.info("  PHASE 5 EXECUTION SUMMARY")
    log.info("  Unique Entities Embedded        : %d", n_unique_entities)
    log.info("  Embedding Dimension             : %d", name_embs.shape[1])
    log.info("  Total Runtime                   : %.2f s", total_time)
    log.info("  Peak Memory Usage               : %.1f MB", peak_mem)
    log.info("  Output Parquet File Size        : %.2f MB", file_size_mb)
    log.info("═════════════════════════════════════════════════════════════")

    return n_unique_entities, name_embs.shape[1], total_time, peak_mem, file_size_mb


if __name__ == "__main__":
    run_embeddings_pipeline()
