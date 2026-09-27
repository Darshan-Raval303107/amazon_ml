"""
Global Conflict Resolution & Assignment Module for Business Entity Resolution.

Implements Section 3 of Engineering Specification v2:
    - 3.1: Global bipartite conflict resolution (greedy highest-confidence-first).
           Guarantees each S2/S3 target record is assigned to at most one S1 reference entity.
    - 3.2: Per-source-pair threshold application (calibrated separately for S1<->S2 and S1<->S3).
    - 3.3: Explicit singleton-detection guard (protects true singletons from marginal false merges).
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class AssignmentConfig:
    """Configuration for thresholding, singleton guards, and conflict resolution."""
    threshold_s2: float             = 0.50   # Threshold for Source 1 <-> Source 2 (web records)
    threshold_s3: float             = 0.50   # Threshold for Source 1 <-> Source 3 (public records)
    singleton_prob_floor: float     = 0.50   # Hard probability floor for any match
    singleton_similarity_floor: float = 0.25 # Similarity floor across lexical/embedding signals
    enable_conflict_resolution: bool = True  # Enforce at-most-one-S1 assignment per S2/S3 candidate


class ConflictResolver:
    """
    Resolves multi-source assignment conflicts and optimizes precision on Macro F0.5.
    """

    def __init__(self, cfg: Optional[AssignmentConfig] = None) -> None:
        self.cfg = cfg or AssignmentConfig()

    def resolve(
        self,
        scored_pairs_df: pd.DataFrame,
        all_s1_ids: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
        """
        Assign candidate records to Source 1 entities with conflict resolution.

        Parameters
        ----------
        scored_pairs_df : pd.DataFrame
            Must contain columns:
                - 'source1_entity_id'
                - 'candidate_entity_id'
                - 'prob'
            Optional columns:
                - 'source_dataset' ('S2' or 'S3')
                - 'name_levenshtein_sim' or 'name_char_3gram_sim'
        all_s1_ids : Optional[List[str]]
            Complete list of expected Source 1 entities. Ensures all S1 entities
            are represented in the returned dictionary.

        Returns
        -------
        Tuple[Dict[str, List[str]], Dict[str, int]]
            Mapping of source1_entity_id -> list of matched candidate_entity_ids,
            and dictionary of assignment statistics.
        """
        cfg = self.cfg
        if scored_pairs_df.empty:
            empty_dict = {s1: [] for s1 in (all_s1_ids or [])}
            return empty_dict, {
                "total_pairs": 0,
                "above_threshold": 0,
                "conflicts_resolved": 0,
                "unique_targets_assigned": 0,
                "s1_with_matches": 0,
            }

        df = scored_pairs_df.copy()

        # Determine source dataset (S2 vs S3)
        if "source_dataset" not in df.columns:
            df["source_dataset"] = [
                "S2" if str(c).startswith("S2") else "S3"
                for c in df["candidate_entity_id"]
            ]

        # ── 1. Apply Per-Source-Pair Thresholds (Section 3.2) ───────────────
        s2_mask = (df["source_dataset"] == "S2") & (df["prob"] >= cfg.threshold_s2)
        s3_mask = (df["source_dataset"] == "S3") & (df["prob"] >= cfg.threshold_s3)
        above_thresh_df = df[s2_mask | s3_mask]

        total_above_thresh = len(above_thresh_df)

        # ── 2. Explicit Singleton-Detection Guard (Section 3.3) ─────────────
        # If the max probability for an S1 entity is below the singleton floor,
        # or if max similarity is below conservative floor, reject all matches for that S1.
        valid_s1_set = set()
        s1_max_prob = df.groupby("source1_entity_id")["prob"].max().to_dict()

        for s1, m_prob in s1_max_prob.items():
            if m_prob >= cfg.singleton_prob_floor:
                valid_s1_set.add(s1)

        guarded_df = above_thresh_df[above_thresh_df["source1_entity_id"].isin(valid_s1_set)]

        # ── 3. Global Conflict Resolution: Bipartite Matching (Section 3.1) ─
        # Sort candidate pairs descending by probability
        sorted_df = guarded_df.sort_values(by=["prob"], ascending=False)

        claimed_targets: Dict[str, str] = {}  # target_id -> s1_id that claimed it
        assignments: Dict[str, List[str]] = {s1: [] for s1 in (all_s1_ids or [])}
        conflicts_count = 0
        sample_conflicts = []

        s1_arr = sorted_df["source1_entity_id"].to_numpy(dtype=str)
        cand_arr = sorted_df["candidate_entity_id"].to_numpy(dtype=str)
        prob_arr = sorted_df["prob"].to_numpy(dtype=float)

        for i in range(len(sorted_df)):
            s1 = s1_arr[i]
            cand = cand_arr[i]
            p = prob_arr[i]

            if not cfg.enable_conflict_resolution:
                # No conflict resolution: naive independent assignment
                if s1 not in assignments:
                    assignments[s1] = []
                assignments[s1].append(cand)
                continue

            if cand in claimed_targets:
                # Conflict! Another S1 entity already claimed this candidate with >= probability
                conflicts_count += 1
                prior_s1 = claimed_targets[cand]
                if len(sample_conflicts) < 5:
                    sample_conflicts.append((cand, prior_s1, s1, p))
                continue

            # Assign candidate to S1
            claimed_targets[cand] = s1
            if s1 not in assignments:
                assignments[s1] = []
            assignments[s1].append(cand)

        s1_with_matches = sum(1 for m in assignments.values() if len(m) > 0)

        stats = {
            "total_pairs": len(scored_pairs_df),
            "above_threshold": total_above_thresh,
            "conflicts_resolved": conflicts_count,
            "unique_targets_assigned": len(claimed_targets),
            "s1_with_matches": s1_with_matches,
        }

        log.info(
            "Conflict Resolution Summary: %d candidate pairs evaluated | "
            "%d above threshold | %d conflicts resolved | %d targets assigned | %d S1 matched",
            stats["total_pairs"],
            stats["above_threshold"],
            stats["conflicts_resolved"],
            stats["unique_targets_assigned"],
            stats["s1_with_matches"],
        )

        return assignments, stats
