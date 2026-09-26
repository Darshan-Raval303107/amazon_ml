"""
Evaluation module for Business Entity Resolution.

Implements the official macro-averaged F_beta (beta = 0.5) metric:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Computes macro F_0.5 per Source 1 entity, averaged across all Source 1 entities,
including singletons (entities with 0 ground truth matches).
Also provides threshold optimization to maximize macro F_0.5.
"""

from typing import Dict, List, Set, Tuple
import numpy as np

from .config import ModelConfig


class Evaluator:
    """Computes competition-compliant evaluation metrics and performs threshold tuning."""

    def __init__(self, beta: float = 0.5):
        self.beta = beta
        self.beta_sq = beta ** 2

    def compute_entity_f_score(
        self, predicted_ids: Set[str], ground_truth_ids: Set[str]
    ) -> float:
        """
        Compute F_0.5 for a single Source 1 entity.

        Special Cases (Singletons):
            - If ground truth is empty and prediction is empty: score = 1.0
            - If ground truth is empty and prediction is non-empty: score = 0.0 (false merge)
            - If ground truth is non-empty and prediction is empty: score = 0.0 (missed match)
        """
        # Singleton handling
        if not ground_truth_ids:
            return 1.0 if not predicted_ids else 0.0

        if not predicted_ids:
            return 0.0

        true_positives = len(predicted_ids & ground_truth_ids)
        if true_positives == 0:
            return 0.0

        precision = true_positives / len(predicted_ids)
        recall = true_positives / len(ground_truth_ids)

        # F_0.5 = (1 + 0.5^2) * P * R / (0.5^2 * P + R) = 1.25 * P * R / (0.25 * P + R)
        f_score = ((1 + self.beta_sq) * precision * recall) / (
            self.beta_sq * precision + recall
        )
        return float(f_score)

    def compute_macro_f_score(
        self,
        predictions: Dict[str, Set[str]],
        ground_truth: Dict[str, Set[str]],
    ) -> float:
        """
        Compute macro-averaged F_0.5 across all Source 1 entities in ground_truth.
        """
        scores = []
        for s1_id, true_set in ground_truth.items():
            pred_set = predictions.get(s1_id, set())
            scores.append(self.compute_entity_f_score(pred_set, true_set))

        return float(np.mean(scores)) if scores else 0.0

    def find_optimal_threshold(
        self,
        candidate_probabilities: Dict[str, List[Tuple[str, float]]],
        ground_truth: Dict[str, Set[str]],
        threshold_range: Tuple[float, float, int] = (0.1, 0.9, 81),
    ) -> Tuple[float, float]:
        """
        Grid search for the decision threshold that maximizes macro F_0.5 on validation data.

        Returns:
            Tuple of (best_threshold, best_macro_f05_score)
        """
        low, high, steps = threshold_range
        thresholds = np.linspace(low, high, steps)
        best_threshold = 0.50
        best_score = -1.0

        for thresh in thresholds:
            pred_dict: Dict[str, Set[str]] = {}
            for s1_id, cands in candidate_probabilities.items():
                matched = {cand_id for cand_id, prob in cands if prob >= thresh}
                pred_dict[s1_id] = matched

            score = self.compute_macro_f_score(pred_dict, ground_truth)
            if score > best_score:
                best_score = score
                best_threshold = float(thresh)

        return best_threshold, best_score
