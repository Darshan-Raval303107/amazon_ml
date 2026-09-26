"""
Data Analysis & EDA module for Business Entity Resolution.

Performs complete exploratory data analysis across all training and test datasets:
  1. File existence verification
  2. Row counts (fast line-count, not full load)
  3. Column names, data types
  4. Missing value analysis (business_name, business_address, country)
  5. Unique country distributions
  6. Duplicate analysis (sampled)
  7. Ground-truth match distribution analysis (full load - compact file)
  8. Text length and token count statistics (sampled)
  9. Noise pattern detection with real examples (sampled)
 10. Blocking strategy recommendation based solely on observed data

Strategy: Full load only for compact files (ground_truth ~127 MB).
          All source files (~175-509 MB each) use chunked row counting
          + sampled loading (50k rows) for stats/noise analysis.
          This keeps the script under ~2 min on any machine.

Usage (from project root):
    python -X utf8 code/business_entity_resolution/src/data_analysis.py
"""

from __future__ import annotations

import io
import re
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Force UTF-8 stdout early (before any prints) ────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Suppress pandas regex-group UserWarnings (cosmetic only)
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

import numpy as np
import pandas as pd

# ─── Resolve project root ─────────────────────────────────────────────────────
_SRC_DIR      = Path(__file__).resolve().parent           # .../src
_PROJECT_ROOT = _SRC_DIR.parent.parent.parent             # .../business-entity-resolution

try:
    from .config import PathConfig
    _P = PathConfig()
    TRAIN_DIR  = _P.train_dir
    TEST_DIR   = _P.test_dir
    OUTPUT_DIR = _P.output_dir
except ImportError:
    TRAIN_DIR  = _PROJECT_ROOT / "dataset" / "train"
    TEST_DIR   = _PROJECT_ROOT / "dataset" / "test"
    OUTPUT_DIR = _PROJECT_ROOT / "output"

# ─── File manifest ────────────────────────────────────────────────────────────
TRAIN_FILES: Dict[str, Path] = {
    "train_source1":       TRAIN_DIR / "train_source1.tsv",
    "train_source2":       TRAIN_DIR / "train_source2.tsv",
    "train_source3":       TRAIN_DIR / "train_source3.tsv",
    "train_ground_truth":  TRAIN_DIR / "train_ground_truth.tsv",
}
TEST_FILES: Dict[str, Path] = {
    "test_source1": TEST_DIR / "test_source1.tsv",
    "test_source2": TEST_DIR / "test_source2.tsv",
    "test_source3": TEST_DIR / "test_source3.tsv",
}

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]

# Number of rows to sample for stats / noise detection on large files
SAMPLE_ROWS = 50_000

# Legal suffix groups for noise detection
LEGAL_SUFFIX_VARIANTS: Dict[str, List[str]] = {
    "private_limited": ["pvt ltd", "private limited", "pvt. ltd.", "pvt.ltd"],
    "limited":         ["ltd", "ltd.", "limited"],
    "corporation":     ["corp", "corp.", "corporation"],
    "incorporated":    ["inc", "inc.", "incorporated"],
    "llc":             ["llc", "l.l.c."],
    "company":         [r"co\.", "co", "company"],
    "enterprises":     ["enterprises", r"ent\."],
}

ADDRESS_ABBR_PAIRS: List[Tuple[str, str]] = [
    ("rd",   "road"),
    ("st",   "street"),
    ("ave",  "avenue"),
    ("blvd", "boulevard"),
]


# ─── Utilities ────────────────────────────────────────────────────────────────

def _sep() -> None:
    print("\n" + "=" * 70)

def _section(title: str) -> None:
    _sep()
    print(f"  {title}")
    print("=" * 70)

def _fmt_pct(n: int, total: int) -> str:
    if total == 0:
        return "0 (0.0%)"
    return f"{n:,} ({100 * n / total:.1f}%)"

def _md_table(headers: List[str], rows: List[List[Any]]) -> str:
    col_w = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
             for i, h in enumerate(headers)]
    sep  = "| " + " | ".join("-" * w for w in col_w) + " |"
    head = "| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, col_w)) + " |"
    body = ["| " + " | ".join(str(r[i]).ljust(col_w[i]) for i in range(len(headers))) + " |"
            for r in rows]
    return "\n".join([head, sep] + body)

def _fast_line_count(path: Path) -> int:
    """Count non-header data lines quickly without loading into pandas."""
    count = 0
    with open(path, "rb") as f:
        f.readline()  # skip header
        for chunk in iter(lambda: f.read(1 << 20), b""):
            count += chunk.count(b"\n")
    return count

def _has_non_ascii(s: str) -> bool:
    return any(ord(c) > 127 for c in s)

def _token_count(s: Any) -> int:
    return len(str(s).split()) if pd.notna(s) else 0

def _char_len(s: Any) -> int:
    return len(str(s)) if pd.notna(s) else 0


# ─── 1. File Verification ─────────────────────────────────────────────────────

def verify_files() -> Tuple[Dict[str, Path], List[str]]:
    all_files = {**TRAIN_FILES, **TEST_FILES}
    found:   Dict[str, Path] = {}
    missing: List[str]       = []
    for name, path in all_files.items():
        if path.is_file():
            found[name] = path
        else:
            missing.append(str(path))
    return found, missing


# ─── 2. Sample-based source loader ────────────────────────────────────────────

def load_sample(path: Path, nrows: int = SAMPLE_ROWS) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, nrows=nrows,
                     keep_default_na=False, na_values=[""])
    for col in SOURCE_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[SOURCE_COLS]


def load_full(path: Path) -> pd.DataFrame:
    """Load full file (used only for compact ground truth)."""
    return pd.read_csv(path, sep="\t", dtype=str,
                       keep_default_na=False, na_values=[""])


# ─── 3. Missing-value profile (on sample) ────────────────────────────────────

def profile_missing(df: pd.DataFrame) -> Dict[str, Any]:
    n = len(df)
    result: Dict[str, Any] = {"sample_rows": n}
    for col in ["business_name", "business_address", "country"]:
        if col in df.columns:
            null_c = int(df[col].isna().sum())
            result[f"miss_{col}"]     = null_c
            result[f"miss_{col}_pct"] = round(100 * null_c / n, 2) if n else 0.0
    return result


# ─── 4. Country distribution (on sample) ─────────────────────────────────────

def country_dist(df: pd.DataFrame) -> Dict[str, int]:
    if "country" not in df.columns:
        return {}
    return df["country"].fillna("MISSING").value_counts().to_dict()


# ─── 5. Duplicate analysis (on sample) ───────────────────────────────────────

def analyze_duplicates(df: pd.DataFrame) -> Dict[str, int]:
    n = len(df)
    r: Dict[str, int] = {"sample_rows": n}
    r["exact_dup_rows"]          = int(df.duplicated().sum())
    r["dup_entity_id"]           = int(df["entity_id"].duplicated().sum()) if "entity_id" in df.columns else 0
    clean_name = df["business_name"].str.strip().str.lower()
    clean_addr = df["business_address"].str.strip().str.lower()
    r["dup_business_name"]       = int(clean_name.duplicated().sum())
    r["dup_business_address"]    = int(clean_addr.duplicated().sum())
    r["dup_name_address_combo"]  = int((clean_name + "|||" + clean_addr).duplicated().sum())
    return r


# ─── 6. Text statistics (on sample) ──────────────────────────────────────────

def text_stats(series: pd.Series) -> Dict[str, float]:
    clean = series.dropna().astype(str)
    clean = clean[clean.str.strip() != ""]
    if clean.empty:
        return {k: 0.0 for k in
                ["count","mean_chars","median_chars","min_chars","max_chars",
                 "mean_tokens","median_tokens","min_tokens","max_tokens"]}
    cl = clean.apply(_char_len)
    tl = clean.apply(_token_count)
    return {
        "count":         len(clean),
        "mean_chars":    round(float(cl.mean()),   1),
        "median_chars":  round(float(cl.median()), 1),
        "min_chars":     int(cl.min()),
        "max_chars":     int(cl.max()),
        "mean_tokens":   round(float(tl.mean()),   1),
        "median_tokens": round(float(tl.median()), 1),
        "min_tokens":    int(tl.min()),
        "max_tokens":    int(tl.max()),
    }


# ─── 7. Noise Pattern Detection (on sample) ──────────────────────────────────

def detect_noise(df: pd.DataFrame, n_ex: int = 3) -> Dict[str, Any]:
    p: Dict[str, Any] = {}
    names = df["business_name"].dropna().astype(str)
    addrs = df["business_address"].dropna().astype(str)

    # Punctuation in names
    p["punctuation_in_name"] = names[names.str.contains(r"[&.,()\-']+", regex=True, na=False)].head(n_ex).tolist()

    # Legal suffix variants
    suffix_hits: Dict[str, List[str]] = {}
    for canonical, variants in LEGAL_SUFFIX_VARIANTS.items():
        pat = r"\b(" + "|".join(re.escape(v) for v in variants) + r")\b"
        hits = names[names.str.lower().str.contains(pat, regex=True, na=False)].head(n_ex).tolist()
        if hits:
            suffix_hits[canonical] = hits
    p["legal_suffix_variants"] = suffix_hits

    # Address abbreviations
    addr_abbr: Dict[str, List[str]] = {}
    for abbr, full in ADDRESS_ABBR_PAIRS:
        abbr_hits = addrs[addrs.str.lower().str.contains(r"\b" + abbr + r"\b", regex=True, na=False)].head(2).tolist()
        full_hits = addrs[addrs.str.lower().str.contains(r"\b" + full + r"\b", regex=True, na=False)].head(2).tolist()
        if abbr_hits or full_hits:
            addr_abbr[f"{abbr}/{full}"] = abbr_hits[:2] + full_hits[:2]
    p["address_abbreviations"] = addr_abbr

    # Transliteration / non-ASCII
    p["transliteration_names"]     = names[names.apply(_has_non_ascii)].head(n_ex).tolist()
    p["transliteration_addresses"] = addrs[addrs.apply(_has_non_ascii)].head(n_ex).tolist()

    # Landmark-based addresses
    lm_pat = r"\b(near|opp|opposite|behind|beside|next to)\b"
    p["landmark_addresses"] = addrs[addrs.str.lower().str.contains(lm_pat, regex=True, na=False)].head(n_ex).tolist()

    # Domain names used as business name
    dom_pat = r"\.(com|in|org|net|co\.in)"
    p["domain_as_business_name"] = names[names.str.lower().str.contains(dom_pat, regex=True, na=False)].head(n_ex).tolist()

    # PIN/ZIP codes in addresses
    pin_pat = r"\b\d{5,6}\b"
    p["pin_zip_in_address"] = addrs[addrs.str.contains(pin_pat, regex=True, na=False)].head(n_ex).tolist()

    # Very short addresses
    p["very_short_addresses"] = addrs[addrs.str.strip().str.len() < 10].head(n_ex).tolist()

    # All-caps names
    p["all_caps_names"] = names[(names.str.upper() == names) & (names.str.len() > 3)].head(n_ex).tolist()

    return p


# ─── 8. Ground Truth Analysis (full load) ────────────────────────────────────

def analyze_ground_truth(gt_df: pd.DataFrame) -> Dict[str, Any]:
    def parse_ids(raw) -> List[str]:
        if pd.isna(raw) or str(raw).strip() == "":
            return []
        return [x.strip() for x in str(raw).split(",") if x.strip()]

    gt = gt_df.copy()
    gt["_ids"]       = gt["matched_entity_ids"].apply(parse_ids)
    gt["_n_matches"] = gt["_ids"].apply(len)

    total = len(gt)
    s: Dict[str, Any] = {
        "total_s1_entities": total,
        "singletons":        int((gt["_n_matches"] == 0).sum()),
        "one_match":         int((gt["_n_matches"] == 1).sum()),
        "multi_match":       int((gt["_n_matches"] >= 2).sum()),
        "avg_matches":       round(float(gt["_n_matches"].mean()), 3),
        "median_matches":    float(gt["_n_matches"].median()),
        "max_matches":       int(gt["_n_matches"].max()),
    }
    all_ids      = [m for ids in gt["_ids"] for m in ids]
    s["total_matched_ids"] = len(all_ids)
    s["s2_matched_ids"]    = sum(1 for m in all_ids if m.startswith("S2-"))
    s["s3_matched_ids"]    = sum(1 for m in all_ids if m.startswith("S3-"))
    dist = Counter(gt["_n_matches"].tolist())
    s["match_count_distribution"] = {k: dist[k] for k in sorted(dist)}
    return s


# ─── 9. Main Orchestrator ─────────────────────────────────────────────────────

def run_analysis() -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    # -- Step 1: File Verification --
    _section("STEP 1: File Verification")
    found, missing = verify_files()
    results["found_files"]   = list(found.keys())
    results["missing_files"] = missing

    total_expected = len(TRAIN_FILES) + len(TEST_FILES)
    print(f"  Found : {len(found)} / {total_expected} expected files")
    for name, path in found.items():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"    [OK]      {name:25s}  {size_mb:8.1f} MB")
    if missing:
        print("\n  MISSING FILES (cannot compute these):")
        for m in missing:
            print(f"    [MISSING] {m}")

    if not found:
        print("  No files found -- aborting.")
        return results

    # -- Step 2: Row counts (fast, no full load) --
    _section("STEP 2: Row Counts (exact, via line counting)")
    row_counts: Dict[str, int] = {}
    for name, path in found.items():
        n = _fast_line_count(path)
        row_counts[name] = n
        print(f"    {name:30s}: {n:>12,} rows")
    results["row_counts"] = row_counts

    # -- Steps 3-7: Profile each source file using a sample --
    source_profiles:  Dict[str, Any] = {}
    country_dists:    Dict[str, Any] = {}
    duplicate_stats:  Dict[str, Any] = {}
    text_statistics:  Dict[str, Any] = {}
    noise_patterns:   Dict[str, Any] = {}

    source_targets = [
        ("train_source1", TRAIN_FILES.get("train_source1"), "train"),
        ("train_source2", TRAIN_FILES.get("train_source2"), "train"),
        ("train_source3", TRAIN_FILES.get("train_source3"), "train"),
        ("test_source1",  TEST_FILES.get("test_source1"),  "test"),
        ("test_source2",  TEST_FILES.get("test_source2"),  "test"),
        ("test_source3",  TEST_FILES.get("test_source3"),  "test"),
    ]

    for name, path, split in source_targets:
        if path is None or not path.is_file():
            print(f"\n  [SKIP] {name} -- file not found.")
            continue

        _section(f"STEP 3-7: Profiling  {name}  (sample={SAMPLE_ROWS:,} rows)")
        df = load_sample(path, nrows=SAMPLE_ROWS)
        total_rows = row_counts.get(name, len(df))
        print(f"  Total rows (exact)   : {total_rows:,}")
        print(f"  Sample rows loaded   : {len(df):,}")
        print(f"  Columns              : {list(df.columns)}")
        print(f"  Data types           : {df.dtypes.to_dict()}")

        # Missing values
        miss = profile_missing(df)
        source_profiles[name] = {**miss, "total_rows": total_rows}
        print(f"\n  MISSING VALUES (sample-based):")
        for col in ["business_name", "business_address", "country"]:
            print(f"    {col:25s}: {miss.get(f'miss_{col}',0):>6,}  ({miss.get(f'miss_{col}_pct',0):.2f}%)")

        # Country distribution
        cd = country_dist(df)
        country_dists[name] = cd
        print(f"\n  COUNTRY DISTRIBUTION (sample-based):")
        sample_n = len(df)
        for country, cnt in sorted(cd.items(), key=lambda x: -x[1]):
            print(f"    {country:20s}: {cnt:>8,}  ({100*cnt/sample_n:.1f}%)")

        # Duplicates
        dups = analyze_duplicates(df)
        duplicate_stats[name] = {**dups, "total_rows": total_rows}
        n = dups["sample_rows"]
        print(f"\n  DUPLICATE ANALYSIS (sample-based):")
        print(f"    Exact dup rows       : {dups['exact_dup_rows']:>8,}  ({100*dups['exact_dup_rows']/n:.2f}%)")
        print(f"    Dup entity_id        : {dups['dup_entity_id']:>8,}")
        print(f"    Dup business_name    : {dups.get('dup_business_name',0):>8,}  ({100*dups.get('dup_business_name',0)/n:.2f}%)")
        print(f"    Dup address          : {dups.get('dup_business_address',0):>8,}  ({100*dups.get('dup_business_address',0)/n:.2f}%)")
        print(f"    Dup name+addr combo  : {dups.get('dup_name_address_combo',0):>8,}  ({100*dups.get('dup_name_address_combo',0)/n:.2f}%)")

        # Text statistics
        ns = text_stats(df["business_name"])
        as_ = text_stats(df["business_address"])
        text_statistics[name] = {"name": ns, "address": as_}
        print(f"\n  TEXT STATS -- business_name:")
        print(f"    chars  avg={ns['mean_chars']:.1f}  med={ns['median_chars']:.1f}  min={ns['min_chars']}  max={ns['max_chars']}")
        print(f"    tokens avg={ns['mean_tokens']:.1f}  med={ns['median_tokens']:.1f}  min={ns['min_tokens']}  max={ns['max_tokens']}")
        print(f"  TEXT STATS -- business_address:")
        print(f"    chars  avg={as_['mean_chars']:.1f}  med={as_['median_chars']:.1f}  min={as_['min_chars']}  max={as_['max_chars']}")
        print(f"    tokens avg={as_['mean_tokens']:.1f}  med={as_['median_tokens']:.1f}  min={as_['min_tokens']}  max={as_['max_tokens']}")

        # Noise patterns (training only -- no GT for test)
        if split == "train":
            print(f"\n  NOISE PATTERNS (sample-based):")
            np_ = detect_noise(df)
            noise_patterns[name] = np_

            if np_.get("legal_suffix_variants"):
                print(f"    Legal suffix variants:")
                for canon, exs in np_["legal_suffix_variants"].items():
                    print(f"      [{canon}]  e.g. {exs[0] if exs else 'n/a'}")

            if np_.get("transliteration_names"):
                print(f"    Non-ASCII (transliterated) names:")
                for ex in np_["transliteration_names"][:2]:
                    print(f"      -> {ex[:80]}")

            if np_.get("landmark_addresses"):
                print(f"    Landmark-based addresses:")
                for ex in np_["landmark_addresses"][:2]:
                    print(f"      -> {ex[:80]}")

            if np_.get("domain_as_business_name"):
                print(f"    Domain names as business name:")
                for ex in np_["domain_as_business_name"][:2]:
                    print(f"      -> {ex[:80]}")

            if np_.get("pin_zip_in_address"):
                print(f"    PIN/ZIP codes in addresses:")
                for ex in np_["pin_zip_in_address"][:2]:
                    print(f"      -> {ex[:80]}")

            if np_.get("very_short_addresses"):
                print(f"    Very short addresses (<10 chars):")
                for ex in np_["very_short_addresses"][:2]:
                    print(f"      -> '{ex}'")

    results["source_profiles"] = source_profiles
    results["country_dists"]   = country_dists
    results["duplicate_stats"] = duplicate_stats
    results["text_statistics"] = text_statistics
    results["noise_patterns"]  = noise_patterns

    # -- Step 8: Ground Truth Analysis (full load) --
    gt_stats: Dict[str, Any] = {}
    gt_path = TRAIN_FILES.get("train_ground_truth")
    if gt_path and gt_path.is_file():
        _section("STEP 8: Ground Truth Analysis (full load)")
        print("  Loading train_ground_truth.tsv ...")
        gt_df    = load_full(gt_path)
        gt_stats = analyze_ground_truth(gt_df)
        total    = gt_stats["total_s1_entities"]

        print(f"  Total S1 entities        : {total:,}")
        print(f"  Singletons (0 matches)   : {_fmt_pct(gt_stats['singletons'],  total)}")
        print(f"  One-match entities       : {_fmt_pct(gt_stats['one_match'],   total)}")
        print(f"  Multi-match (>=2)        : {_fmt_pct(gt_stats['multi_match'], total)}")
        print(f"  Avg matches per S1       : {gt_stats['avg_matches']:.3f}")
        print(f"  Median matches per S1    : {gt_stats['median_matches']}")
        print(f"  Max matches (one entity) : {gt_stats['max_matches']:,}")
        print(f"  Total matched IDs (S2+S3): {gt_stats['total_matched_ids']:,}")
        print(f"    -> from Source 2       : {gt_stats['s2_matched_ids']:,}")
        print(f"    -> from Source 3       : {gt_stats['s3_matched_ids']:,}")
        print(f"\n  Match-count distribution (top 10 buckets):")
        for k, v in list(gt_stats["match_count_distribution"].items())[:10]:
            bar = "#" * min(int(v / max(gt_stats["match_count_distribution"].values()) * 35), 35)
            print(f"    {k:3d} match(es): {v:>9,}  {bar}")
    results["gt_stats"] = gt_stats

    return results


# ─── 10. Report Generator ─────────────────────────────────────────────────────

def generate_report(results: Dict[str, Any]) -> Path:
    out = OUTPUT_DIR / "dataset_analysis_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    sp  = results.get("source_profiles", {})
    cd  = results.get("country_dists",   {})
    ds  = results.get("duplicate_stats", {})
    ts  = results.get("text_statistics", {})
    np_ = results.get("noise_patterns",  {})
    gt  = results.get("gt_stats",        {})
    rc  = results.get("row_counts",      {})

    L: List[str] = []
    a = L.append

    a("# Dataset Analysis Report -- Business Entity Resolution")
    a("")
    a("> Auto-generated by `code/business_entity_resolution/src/data_analysis.py`  ")
    a(f"> Statistics based on exact row counts and {SAMPLE_ROWS:,}-row samples per source file.")
    a("")
    a("---")
    a("")

    # 1. Dataset Summary
    a("## 1. Dataset Summary")
    a("")
    rows = []
    for name in ["train_source1","train_source2","train_source3",
                 "test_source1","test_source2","test_source3"]:
        split = "Train" if "train" in name else "Test"
        total = rc.get(name, "?")
        prof  = sp.get(name, {})
        rows.append([name, split,
                     f"{total:,}" if isinstance(total, int) else total,
                     f"{prof.get('miss_business_name_pct',0):.1f}%",
                     f"{prof.get('miss_business_address_pct',0):.1f}%",
                     f"{prof.get('miss_country_pct',0):.1f}%"])
    a(_md_table(["File","Split","Total Rows","Miss Name","Miss Addr","Miss Country"], rows))
    a("")

    # 2. Missing Values
    a("## 2. Missing Value Analysis  *(sample-based)*")
    a("")
    mv_rows = []
    for name, prof in sp.items():
        mv_rows.append([
            name,
            f"{prof.get('miss_business_name',0):,}",
            f"{prof.get('miss_business_name_pct',0):.2f}%",
            f"{prof.get('miss_business_address',0):,}",
            f"{prof.get('miss_business_address_pct',0):.2f}%",
            f"{prof.get('miss_country',0):,}",
            f"{prof.get('miss_country_pct',0):.2f}%",
        ])
    a(_md_table(["Source","Miss Name","  %","Miss Addr","  %","Miss Country","  %"], mv_rows))
    a("")

    # 3. Country Distribution
    a("## 3. Country Distribution  *(sample-based)*")
    a("")
    for name, dist in cd.items():
        n = sum(dist.values())
        a(f"### {name}")
        cr = [[c, f"{cnt:,}", f"{100*cnt/n:.1f}%"] for c, cnt in sorted(dist.items(), key=lambda x: -x[1])]
        a(_md_table(["Country","Count","Pct"], cr))
        a("")

    # 4. Duplicate Analysis
    a("## 4. Duplicate Analysis  *(sample-based)*")
    a("")
    dup_rows = []
    for name, d in ds.items():
        n = d["sample_rows"]
        dup_rows.append([
            name,
            f"{d['exact_dup_rows']:,}", f"{100*d['exact_dup_rows']/n:.2f}%",
            f"{d.get('dup_business_name',0):,}", f"{100*d.get('dup_business_name',0)/n:.2f}%",
            f"{d.get('dup_business_address',0):,}", f"{100*d.get('dup_business_address',0)/n:.2f}%",
            f"{d.get('dup_name_address_combo',0):,}", f"{100*d.get('dup_name_address_combo',0)/n:.2f}%",
        ])
    a(_md_table(["Source","Exact Dup","%","Dup Name","%","Dup Addr","%","Dup Name+Addr","%"], dup_rows))
    a("")

    # 5. Ground Truth
    a("## 5. Ground Truth Analysis  *(full file)*")
    a("")
    if gt:
        total = gt["total_s1_entities"]
        a("| Metric | Value |")
        a("| :--- | :--- |")
        a(f"| Total S1 entities              | {total:,} |")
        a(f"| Singletons (0 matches)         | {gt['singletons']:,} ({100*gt['singletons']/total:.1f}%) |")
        a(f"| One-match entities             | {gt['one_match']:,} ({100*gt['one_match']/total:.1f}%) |")
        a(f"| Multi-match (>=2 matches)      | {gt['multi_match']:,} ({100*gt['multi_match']/total:.1f}%) |")
        a(f"| Avg matches per S1 entity      | {gt['avg_matches']} |")
        a(f"| Median matches per S1 entity   | {gt['median_matches']} |")
        a(f"| Max matches for one S1 entity  | {gt['max_matches']:,} |")
        a(f"| Total matched IDs (S2+S3)      | {gt['total_matched_ids']:,} |")
        a(f"| Matched from Source-2          | {gt['s2_matched_ids']:,} |")
        a(f"| Matched from Source-3          | {gt['s3_matched_ids']:,} |")
        a("")
        a("### Match-Count Distribution")
        a("")
        dist_rows = [[str(k), f"{v:,}", f"{100*v/total:.2f}%"]
                     for k, v in sorted(gt["match_count_distribution"].items())]
        a(_md_table(["Matches per S1","Count","% of S1 entities"], dist_rows))
        a("")

    # 6. Text Statistics
    a("## 6. Text Length & Token Statistics  *(sample-based)*")
    a("")
    for src, entry in ts.items():
        a(f"### {src}")
        for field_name, st in entry.items():
            a(f"**{field_name}** | chars: avg={st['mean_chars']}, med={st['median_chars']}, "
              f"min={st['min_chars']}, max={st['max_chars']} | "
              f"tokens: avg={st['mean_tokens']}, med={st['median_tokens']}, "
              f"min={st['min_tokens']}, max={st['max_tokens']}")
        a("")

    # 7. Noise Patterns
    a("## 7. Noise Pattern Examples  *(sample-based, training sources)*")
    a("")
    for src, pats in np_.items():
        a(f"### {src}")
        a("")
        if pats.get("legal_suffix_variants"):
            a("**Legal Suffix Variations:**")
            for canon, exs in pats["legal_suffix_variants"].items():
                if exs:
                    a(f"- `{canon}`: e.g. `{exs[0][:80]}`")
            a("")
        if pats.get("transliteration_names"):
            a("**Non-ASCII / Transliterated Names:**")
            for ex in pats["transliteration_names"][:3]:
                a(f"- `{ex[:80]}`")
            a("")
        if pats.get("landmark_addresses"):
            a("**Landmark-Based Addresses:**")
            for ex in pats["landmark_addresses"][:3]:
                a(f"- `{ex[:100]}`")
            a("")
        if pats.get("domain_as_business_name"):
            a("**Domain/URL as Business Name:**")
            for ex in pats["domain_as_business_name"][:3]:
                a(f"- `{ex[:80]}`")
            a("")
        if pats.get("pin_zip_in_address"):
            a("**PIN/ZIP Codes in Addresses:**")
            for ex in pats["pin_zip_in_address"][:3]:
                a(f"- `{ex[:100]}`")
            a("")
        if pats.get("very_short_addresses"):
            a("**Very Short Addresses (<10 chars):**")
            for ex in pats["very_short_addresses"][:3]:
                a(f"- `{ex}`")
            a("")

    # 8. Blocking Strategy
    a("## 8. Recommended Blocking Strategy")
    a("")
    if rc:
        s1n = rc.get("train_source1", 0)
        s2n = rc.get("train_source2", 0)
        s3n = rc.get("train_source3", 0)
        cross = s1n * (s2n + s3n)
        a(f"> Brute-force: {s1n:,} x ({s2n:,} + {s3n:,}) = **{cross:,} candidate pairs** -- infeasible.")
        a("")
    a("### Multi-Pass Blocking Keys (ordered by priority)")
    a("")
    a("| Pass | Key | Rationale |")
    a("| :--- | :--- | :--- |")
    a("| 1 | `country` + first alpha token of normalized name | Eliminates cross-country pairs; highest precision. |")
    a("| 2 | `country` + character-trigram TF-IDF top-k | Handles suffix changes, abbreviations, word-order swaps. |")
    a("| 3 | 6-digit PIN / 5-digit ZIP extracted from address | Strong locality signal where present. |")
    a("| 4 | `country` + name 4-char prefix key | Captures transliteration / OCR noise. |")
    a("| 5 | Soundex of first name token | Catches spelling variants within same language. |")
    a("")
    a("### Design Constraints (from observed data)")
    a("")
    a("- **Country is open-set**: France appears only in test. Never hard-code or one-hot country.")
    a("- **Transliteration**: Devanagari and romanized forms of same entity exist. Apply NFKD normalization.")
    a("- **Legal suffixes**: `pvt ltd`, `private limited`, `pvt. ltd.` must be standardized pre-blocking.")
    a("- **Landmark addresses**: PIN code extraction more reliable than full address matching for Indian records.")
    a("- **Domain names**: Strip `www.`, `.com`, `.in` before building name keys.")
    a("- **Singleton rate**: If high, use conservative (high-precision) threshold to avoid false merges.")
    a("")
    a("## 9. Next Implementation Steps")
    a("")
    a("1. **`normalization.py`**: Legal suffix map, address abbreviation map, NFKD unicode normalization, domain stripping.")
    a("2. **`blocking.py`**: Multi-pass blocking using keys above. Measure recall ceiling and reduction ratio against ground truth.")
    a("3. **`features.py`**: Pairwise Levenshtein, Jaccard, TF-IDF cosine, token overlap, length, country match.")
    a("4. **`embeddings.py`**: MiniLM-L12 dense embeddings for name and address semantic similarity.")
    a("5. **`train.py`**: CatBoost training on labeled pairs; threshold optimize for macro F_0.5.")
    a("")

    out.write_text("\n".join(L), encoding="utf-8")
    print(f"\n  Report written -> {out}")
    return out


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 70)
    print("  Amazon ML Challenge -- Business Entity Resolution")
    print("  Phase 1: Dataset Inspection & Analysis")
    print("=" * 70)

    results = run_analysis()
    generate_report(results)

    print("\n[DONE] Analysis complete.")


if __name__ == "__main__":
    main()
