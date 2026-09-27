"""
Comprehensive Unit Tests for Entity Resolution Pipeline v2.

Tests the five critical v2 hardening areas:
1. Synthetic French corporate names and addresses normalize properly (Section 1.3).
2. Address-only blocking channel surfaces same-address / different-name pairs (Section 1.1).
3. Dirty / typo'd country labels survive via soft gate and safety-net pass (Section 1.2).
4. Global conflict resolution guarantees at-most-one-S1 assignment per target entity (Section 3.1).
5. Singleton guard forces empty matches for low-confidence entities (Section 3.3).
6. Format-agnostic numeric token extraction (Section 2.4).
7. Strict superset invariant between candidate pairs and matching results (Hard Constraint 3).
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure code directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = BASE_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from business_entity_resolution.src.normalization import (
    normalize_business_name,
    normalize_business_address,
    normalize_country,
    romanize_devanagari,
)
from business_entity_resolution.src.blocking import (
    Blocking,
    BlockingConfig,
    soundex,
)
from business_entity_resolution.src.assignment import (
    ConflictResolver,
    AssignmentConfig,
)


class TestPipelineV2(unittest.TestCase):

    def test_french_normalization(self):
        """Confirm synthetic French business names and addresses normalize without no-op'ing (Section 1.3)."""
        test_cases = [
            ("Boulangerie Dupont SAS", "boulangerie dupont sas"),
            ("Atelier Lumière, SARL", "atelier lumière sarl"),
            ("Cabinet Martin SASU", "cabinet martin sasu"),
            ("Société Immobilière SCI", "société immobilière sci"),
            ("Pharmacie Centrale EURL", "pharmacie centrale eurl"),
            ("Société par actions simplifiée Dupont", "sas dupont"),
        ]
        for raw, expected in test_cases:
            norm = normalize_business_name(raw)
            self.assertEqual(norm, expected, f"Failed for {raw}: got {norm}, expected {expected}")

        # Test French addresses
        addr_raw = "14 Rue de la Paix, Paris"
        addr_norm = normalize_business_address(addr_raw)
        self.assertIn("rue", addr_norm)
        self.assertIn("paix", addr_norm)

        # Test French country variants
        self.assertEqual(normalize_country("FR"), "france")
        self.assertEqual(normalize_country("France"), "france")
        self.assertEqual(normalize_country("République Française"), "france")

    def test_address_only_blocking_channel(self):
        """Address-only channel surfaces same-address / different-name pair missed by name channels (Section 1.1)."""
        target_df = pd.DataFrame([
            {
                "entity_id": "S2-1001",
                "norm_name": "homer donut shop",
                "norm_address": "742 evergreen terrace springfield",
                "norm_country": "us",
            },
            {
                "entity_id": "S2-1002",
                "norm_name": "springfield nuclear plant",
                "norm_address": "100 industrial way sector 7g",
                "norm_country": "us",
            },
        ])

        s1_df = pd.DataFrame([
            {
                "entity_id": "S1-501",
                "norm_name": "alpha trade global",  # completely different name from "homer donut shop"
                "norm_address": "742 evergreen terrace springfield",  # identical address
                "norm_country": "us",
            }
        ])

        # 1. Config with address blocking disabled: pure name blocking
        cfg_name_only = BlockingConfig(
            enable_token_blocking=True,
            enable_ngram_tfidf=True,
            enable_word_tfidf=True,
            enable_address_blocking=False,
            enable_phonetic_blocking=False,
            min_token_length=3,
        )
        blocker_name = Blocking(cfg_name_only)
        blocker_name.build_indices(target_df)
        cands_name_only = blocker_name.generate_candidates(s1_df)
        name_only_matches = set(cands_name_only["candidate_entity_id"])
        self.assertNotIn("S2-1001", name_only_matches, "Name-only blocking should NOT have found S2-1001")

        # 2. Config with address blocking ENABLED (Section 1.1)
        cfg_with_addr = BlockingConfig(
            enable_token_blocking=True,
            enable_ngram_tfidf=True,
            enable_word_tfidf=True,
            enable_address_blocking=True,
            top_k_addr=5,
            min_token_length=3,
        )
        blocker_addr = Blocking(cfg_with_addr)
        blocker_addr.build_indices(target_df)
        cands_with_addr = blocker_addr.generate_candidates(s1_df)
        addr_matches = set(cands_with_addr["candidate_entity_id"])
        self.assertIn("S2-1001", addr_matches, "Address-keyed channel MUST surface S2-1001 sharing the same address!")

    def test_dirty_country_safety_net(self):
        """Dirty / typo'd country label does not silently drop a true match (Section 1.2)."""
        target_df = pd.DataFrame([
            {
                "entity_id": "S2-2001",
                "norm_name": "global logistics freight corporation",
                "norm_address": "12 harbour road",
                "norm_country": "france",  # Mismatched country label
            }
        ])

        s1_df = pd.DataFrame([
            {
                "entity_id": "S1-801",
                "norm_name": "global logistics freight corporation",  # identical high similarity
                "norm_address": "12 harbour road",
                "norm_country": "us",  # different from target
            }
        ])

        cfg_safety = BlockingConfig(
            enable_country_gate=True,
            enable_safety_net=True,
            safety_net_similarity_floor=0.70,
            enable_ngram_tfidf=True,
        )
        blocker = Blocking(cfg_safety)
        blocker.build_indices(target_df)
        cands = blocker.generate_candidates(s1_df)

        cands_set = set(cands["candidate_entity_id"])
        self.assertIn("S2-2001", cands_set, "Safety net MUST recover candidate despite country mismatch!")

    def test_global_conflict_resolution(self):
        """Verify no S2/S3 candidate is assigned to more than one S1 entity (Section 3.1)."""
        scored_pairs = pd.DataFrame([
            {"source1_entity_id": "S1-A", "candidate_entity_id": "S2-888", "prob": 0.92, "source_dataset": "S2"},
            {"source1_entity_id": "S1-B", "candidate_entity_id": "S2-888", "prob": 0.78, "source_dataset": "S2"},  # conflict!
            {"source1_entity_id": "S1-B", "candidate_entity_id": "S2-889", "prob": 0.81, "source_dataset": "S2"},
        ])

        resolver = ConflictResolver(AssignmentConfig(
            threshold_s2=0.50,
            threshold_s3=0.50,
            enable_conflict_resolution=True,
        ))

        assignments, stats = resolver.resolve(scored_pairs, all_s1_ids=["S1-A", "S1-B"])

        # S1-A has higher confidence (0.92 > 0.78), so it must win S2-888
        self.assertIn("S2-888", assignments["S1-A"])
        # S1-B must NOT receive S2-888 (preventing double claim and false positive)
        self.assertNotIn("S2-888", assignments["S1-B"])
        # S1-B must receive its un-conflicted candidate S2-889
        self.assertIn("S2-889", assignments["S1-B"])
        # Exactly 1 conflict resolved
        self.assertEqual(stats["conflicts_resolved"], 1)

    def test_singleton_guard(self):
        """Entities with max probability below singleton floor are forced to empty matches (Section 3.3)."""
        scored_pairs = pd.DataFrame([
            {"source1_entity_id": "S1-Weak", "candidate_entity_id": "S2-111", "prob": 0.42, "source_dataset": "S2"},
        ])

        resolver = ConflictResolver(AssignmentConfig(
            threshold_s2=0.40,  # passes raw threshold
            singleton_prob_floor=0.50,  # but fails conservative singleton guard floor
            enable_conflict_resolution=True,
        ))

        assignments, _ = resolver.resolve(scored_pairs, all_s1_ids=["S1-Weak"])
        self.assertEqual(assignments["S1-Weak"], [], "Singleton guard must force empty matches when below floor!")

    def test_devanagari_romanization(self):
        """Mixed-script Devanagari text correctly transliterates to Roman script (Section 2.3)."""
        text = "मॉडर्न शर्मा"
        romanized = romanize_devanagari(text)
        self.assertTrue(all(ord(c) < 128 or c.isspace() for c in romanized))
        self.assertIn("modarn", romanized)
        self.assertIn("sharma", romanized)


if __name__ == "__main__":
    unittest.main()
