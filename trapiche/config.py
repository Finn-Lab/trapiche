from __future__ import annotations
from typing import Optional, List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrapicheWorkflowParams(BaseSettings):
    """Parameters for the Trapiche workflow.

    All fields can be overridden via environment variables using the
    `TRAPICHE_` prefix (e.g. `TRAPICHE_RUN_TEXT=false`).
    """

    model_config = SettingsConfigDict(env_prefix="TRAPICHE_", case_sensitive=False)

    run_text: bool = True
    keep_text_results: bool = True
    run_vectorise: bool = True
    keep_vectorise_results: bool = False
    run_taxonomy: bool = True
    keep_taxonomy_results: bool = True
    output_keys: Optional[List[str]] = Field(
        default_factory=lambda: [
            "text_predictions",
            "raw_unambiguous_prediction",
            "constrained_unambiguous_prediction",
            "final_selected_prediction",
            "final_selected_prediction_GOLD",
        ]
    )
    sample_study_text_heuristic: bool = False


class TextToBiomeParams(BaseSettings):
    """Configuration parameters for text biome prediction.

    Overridable via environment variables with the `TRAPICHE_` prefix, e.g.:
    TRAPICHE_HF_MODEL, TRAPICHE_DEVICE, etc.
    """

    model_config = SettingsConfigDict(env_prefix="TRAPICHE_", case_sensitive=False)

    device: Optional[str] = None
    max_length: int = 256
    threshold_rule: float | int | str = 0.01
    split_sentences: bool = False
    hf_model: str = "SantiagoSanchezF/trapiche-biome-classifier-text"
    model_version: str = "1.0"


class TaxonomyToVectorParams(BaseSettings):
    """Configuration parameters for taxonomy vectorization.
    """

    model_config = SettingsConfigDict(env_prefix="TRAPICHE_", case_sensitive=False)

    hf_model: str = "SantiagoSanchezF/trapiche-biome-vectorizer-taxonomy"
    model_version: str = "1.0"


class TaxonomyToBiomeParams(BaseSettings):
    """Configuration parameters for deep lineage prediction.

    Overridable via environment variables with the `TRAPICHE_` prefix.
    """

    model_config = SettingsConfigDict(env_prefix="TRAPICHE_", case_sensitive=False)

    batch_size: int = 200
    dominance_threshold: float = 0.5
    top_prob_diff_threshold: float = 0.05
    top_prob_ratio_threshold: float = 0.9
    hf_model: str = "SantiagoSanchezF/trapiche-biome-classifier-taxonomy"
    model_version: str = "1.0"


"""LOGGING CONFIGURATION
"""

import logging
import logging.handlers
from pathlib import Path
import sys
from typing import Optional


def setup_logging(logfile: Optional[str] = None, level: int = logging.INFO) -> None:
    """Configure logging for CLI (file) or API (stdout)."""
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(name)s [%(levelname)s] %(message)s")

    if logfile:
        path = Path(logfile)
        if path.parent and not str(path.parent) == ".":
            path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(str(path), maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(formatter)
    logger.addHandler(handler)