"""
Configuration module for Business Entity Resolution pipeline.

Defines global parameters, filesystem paths, exact TSV headers,
model architectures, and hyperparameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class PathConfig:
    """Project filesystem paths."""
    # Base directories (relative to repository root or execution context)
    base_dir: Path = Path(__file__).resolve().parent.parent.parent.parent
    dataset_dir: Path = base_dir / "dataset"
    train_dir: Path = dataset_dir / "train"
    test_dir: Path = dataset_dir / "test"
    output_dir: Path = base_dir / "output"
    models_dir: Path = base_dir / "code" / "business_entity_resolution" / "models"
    experiments_dir: Path = base_dir / "code" / "business_entity_resolution" / "experiments"
    utils_dir: Path = base_dir / "utils"
    artifacts_dir: Path = models_dir

    # Input file names
    train_s1_file: str = "train_source1.tsv"
    train_s2_file: str = "train_source2.tsv"
    train_s3_file: str = "train_source3.tsv"
    train_ground_truth_file: str = "train_ground_truth.tsv"

    test_s1_file: str = "test_source1.tsv"
    test_s2_file: str = "test_source2.tsv"
    test_s3_file: str = "test_source3.tsv"

    # Submission output file names (exact names required by challenge)
    matching_results_file: str = "matching_results.tsv"
    candidate_pairs_file: str = "candidate_pairs.tsv"


@dataclass
class SchemaConfig:
    """TSV schemas compatible with official competition validator."""
    delimiter: str = "\t"
    id_list_delimiter: str = ","

    source_headers: List[str] = field(
        default_factory=lambda: ["entity_id", "business_name", "business_address", "country"]
    )
    ground_truth_headers: List[str] = field(
        default_factory=lambda: ["source1_entity_id", "matched_entity_ids"]
    )
    candidate_headers: List[str] = field(
        default_factory=lambda: ["source1_entity_id", "candidate_entity_ids"]
    )
    matching_headers: List[str] = field(
        default_factory=lambda: ["source1_entity_id", "matched_entity_ids"]
    )


@dataclass
class ModelConfig:
    """Model architectures and feature extraction settings."""
    # Deep Learning: Multilingual Sentence Transformer
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_batch_size: int = 128
    max_sequence_length: int = 128

    # Machine Learning: CatBoost Classifier
    catboost_iterations: int = 1000
    catboost_learning_rate: float = 0.05
    catboost_depth: int = 6
    random_seed: int = 42

    # Blocking & Candidate Generation
    max_candidates_per_entity: int = 50
    blocking_tfidf_ngram_range: tuple = (2, 3)

    # Evaluation Metric: F_beta with beta = 0.5 (macro-averaged)
    f_beta: float = 0.5
    default_decision_threshold: float = 0.5

