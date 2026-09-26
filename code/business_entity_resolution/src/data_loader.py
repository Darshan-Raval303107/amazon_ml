"""
Data loader module for Business Entity Resolution.

Handles reading and writing of TSV files with strict adherence to the competition schema.
Guarantees explicit tab separation to prevent comma-splitting corruptions.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

from .config import PathConfig, SchemaConfig

log = logging.getLogger(__name__)


class DataLoader:
    """Loads and validates datasets across training and evaluation splits."""

    def __init__(self, path_config: Optional[PathConfig] = None, schema_config: Optional[SchemaConfig] = None):
        self.paths = path_config or PathConfig()
        self.schemas = schema_config or SchemaConfig()

    def load_source_file(self, file_path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load a source TSV file (source 1, 2, or 3) with explicit tab delimiter.

        Expected Columns:
            - entity_id (str): prefixed with S1-, S2-, or S3-
            - business_name (str): business name string
            - business_address (str): address string
            - country (str): country label (US, India, France, etc.)
        """
        df = pd.read_csv(
            str(file_path),
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
        return df

    def load_ground_truth(self, file_path: Path) -> pd.DataFrame:
        """
        Load train_ground_truth.tsv mapping S1 entity IDs to matching S2/S3 IDs.

        Expected Columns:
            - source1_entity_id (str)
            - matched_entity_ids (str): comma-separated string of matched IDs or empty
        """
        df = pd.read_csv(
            str(file_path),
            sep="\t",
            dtype=str,
            keep_default_na=False,
            encoding="utf-8",
            encoding_errors="replace",
        )
        for col in ["source1_entity_id", "matched_entity_ids"]:
            if col not in df.columns:
                df[col] = ""
        return df

    def load_train_dataset(
        self, nrows: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all training sources (source 1, source 2, source 3) and ground truth.
        """
        s1 = self.load_source_file(self.paths.train_dir / self.paths.train_s1_file, nrows=nrows)
        s2 = self.load_source_file(self.paths.train_dir / self.paths.train_s2_file, nrows=nrows)
        s3 = self.load_source_file(self.paths.train_dir / self.paths.train_s3_file, nrows=nrows)
        gt = self.load_ground_truth(self.paths.train_dir / self.paths.train_ground_truth_file)
        return s1, s2, s3, gt

    def load_test_dataset(
        self, nrows: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load test sources (source 1, source 2, source 3).
        """
        s1 = self.load_source_file(self.paths.test_dir / self.paths.test_s1_file, nrows=nrows)
        s2 = self.load_source_file(self.paths.test_dir / self.paths.test_s2_file, nrows=nrows)
        s3 = self.load_source_file(self.paths.test_dir / self.paths.test_s3_file, nrows=nrows)
        return s1, s2, s3

    def save_matching_results(self, predictions: Dict[str, List[str]], output_path: Path) -> None:
        """
        Write matching_results.tsv in the exact required format.

        Rules:
            - Separated by single TAB
            - Header: source1_entity_id\tmatched_entity_ids
            - matched_entity_ids: comma-separated list of S2/S3 IDs, or empty for singletons
            - One row per Source 1 entity in the test set
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(output_path), "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t", lineterminator="\n")
            writer.writerow(["source1_entity_id", "matched_entity_ids"])
            for s1_id, matches in predictions.items():
                seen = set()
                clean = []
                for m in matches:
                    clean_id = m.strip()
                    if clean_id and clean_id not in seen:
                        seen.add(clean_id)
                        clean.append(clean_id)
                writer.writerow([s1_id.strip(), ",".join(clean)])

    def save_candidate_pairs(self, candidates: Dict[str, List[str]], output_path: Path) -> None:
        """
        Write candidate_pairs.tsv in the exact required format.

        Rules:
            - Separated by single TAB
            - Header: source1_entity_id\tcandidate_entity_ids
            - candidate_entity_ids: comma-separated list of candidate S2/S3 IDs
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(output_path), "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t", lineterminator="\n")
            writer.writerow(["source1_entity_id", "candidate_entity_ids"])
            for s1_id, cands in candidates.items():
                seen = set()
                clean = []
                for c in cands:
                    clean_id = c.strip()
                    if clean_id and clean_id not in seen:
                        seen.add(clean_id)
                        clean.append(clean_id)
                writer.writerow([s1_id.strip(), ",".join(clean)])
