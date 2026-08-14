from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def load_config_file(path: str | os.PathLike) -> None:
    """Load model/workflow configuration from a dotenv-style file into the environment.

    The file uses the standard ``KEY=VALUE`` dotenv format (one setting per
    line, `#` comments allowed) with the same ``TRAPICHE_*`` keys documented
    for environment variables (e.g. ``TRAPICHE_TEXT_HF_MODEL``,
    ``TRAPICHE_TAXONOMY_LOCAL_MODEL_DIR``). Values already set in the process
    environment are overridden by the file, mirroring the `--text-hf-model`
    style CLI flags taking precedence when both are provided by the caller.

    Args:
        path: Path to the config/dotenv file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    from dotenv import load_dotenv

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    load_dotenv(config_path, override=True)


class TrapicheWorkflowParams(BaseSettings):
    """Parameters for the Trapiche workflow.

    All fields can be overridden via environment variables using the
    `TRAPICHE_` prefix (e.g. `TRAPICHE_RUN_TEXT=false`).
    """

    model_config = SettingsConfigDict(
        env_prefix="TRAPICHE_", case_sensitive=False, populate_by_name=True
    )

    run_text: bool = True
    keep_text_results: bool = True
    run_vectorise: bool = True
    keep_vectorise_results: bool = False
    run_taxonomy: bool = True
    keep_taxonomy_results: bool = True
    output_keys: list[str] | None = Field(
        default_factory=lambda: [
            "text_predictions",
            "raw_unambiguous_prediction",
            "constrained_unambiguous_prediction",
            "final_selected_prediction",
            "final_selected_prediction_GOLD",
            "_raw_ext_text_pred_project",
            "_raw_ext_text_pred_sample",
        ]
    )
    sample_study_text_heuristic: bool = False
    run_study_summary: bool = True
    study_summary_confidence_threshold: float = 0.2


class TextToBiomeParams(BaseSettings):
    """Configuration parameters for text biome prediction.

    Overridable via environment variables with the `TRAPICHE_` prefix, e.g.:
    TRAPICHE_DEVICE, TRAPICHE_BATCH_SIZE, etc. Model selection fields
    (`hf_model`, `model_version`, `local_model_dir`) use their own
    `TRAPICHE_TEXT_*` variables to avoid colliding with the equivalent fields
    on `TaxonomyToVectorParams`/`TaxonomyToBiomeParams`.
    """

    model_config = SettingsConfigDict(
        env_prefix="TRAPICHE_", case_sensitive=False, populate_by_name=True
    )

    device: str | None = None
    max_length: int = 256
    batch_size: int = 8
    threshold_rule: float | int | str = 0.01
    split_sentences: bool = False
    hf_model: str = Field(
        default="SantiagoSanchezF/trapiche-biome-classifier-text",
        validation_alias=AliasChoices("TRAPICHE_TEXT_HF_MODEL", "TRAPICHE_HF_MODEL"),
    )
    model_version: str = Field(
        default="1.0",
        validation_alias=AliasChoices("TRAPICHE_TEXT_MODEL_VERSION", "TRAPICHE_MODEL_VERSION"),
    )
    local_model_dir: str | None = Field(
        default=None,
        validation_alias=AliasChoices("TRAPICHE_TEXT_LOCAL_MODEL_DIR", "TRAPICHE_LOCAL_MODEL_DIR"),
        description=(
            "Optional local directory mirroring the HF repo layout "
            "(<local_model_dir>/<model_version>/<file>) used instead of "
            "downloading the text model from Hugging Face Hub."
        ),
    )


class TaxonomyToVectorParams(BaseSettings):
    """Configuration parameters for taxonomy vectorization (community2vec).

    Model selection fields use `TRAPICHE_VECTOR_*` environment variables.
    """

    model_config = SettingsConfigDict(
        env_prefix="TRAPICHE_", case_sensitive=False, populate_by_name=True
    )

    hf_model: str = Field(
        default="SantiagoSanchezF/trapiche-biome-vectorizer-taxonomy",
        validation_alias=AliasChoices("TRAPICHE_VECTOR_HF_MODEL", "TRAPICHE_HF_MODEL"),
    )
    model_version: str = Field(
        default="1.0",
        validation_alias=AliasChoices("TRAPICHE_VECTOR_MODEL_VERSION", "TRAPICHE_MODEL_VERSION"),
    )
    local_model_dir: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "TRAPICHE_VECTOR_LOCAL_MODEL_DIR", "TRAPICHE_LOCAL_MODEL_DIR"
        ),
        description=(
            "Optional local directory mirroring the HF repo layout "
            "(<local_model_dir>/<model_version>/<file>) used instead of "
            "downloading the community2vec assets from Hugging Face Hub."
        ),
    )


class TaxonomyToBiomeParams(BaseSettings):
    """Configuration parameters for deep lineage prediction.

    Overridable via environment variables with the `TRAPICHE_` prefix. Model
    selection fields use `TRAPICHE_TAXONOMY_*` environment variables.
    """

    model_config = SettingsConfigDict(
        env_prefix="TRAPICHE_", case_sensitive=False, populate_by_name=True
    )

    batch_size: int = 200
    dominance_threshold: float = 0.5
    top_prob_diff_threshold: float = 0.05
    top_prob_ratio_threshold: float = 0.9
    hf_model: str = Field(
        default="SantiagoSanchezF/trapiche-biome-classifier-taxonomy",
        validation_alias=AliasChoices("TRAPICHE_TAXONOMY_HF_MODEL", "TRAPICHE_HF_MODEL"),
    )
    model_version: str = Field(
        default="1.0",
        validation_alias=AliasChoices("TRAPICHE_TAXONOMY_MODEL_VERSION", "TRAPICHE_MODEL_VERSION"),
    )
    local_model_dir: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "TRAPICHE_TAXONOMY_LOCAL_MODEL_DIR", "TRAPICHE_LOCAL_MODEL_DIR"
        ),
        description=(
            "Optional local directory mirroring the HF repo layout "
            "(<local_model_dir>/<model_version>/<file>) used instead of "
            "downloading the taxonomy classifier model from Hugging Face Hub."
        ),
    )


def setup_logging(logfile: str | None = None, level: int = logging.INFO) -> None:
    """Configure logging for CLI (file) or API (stdout)."""
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(name)s [%(levelname)s] %(message)s")

    if logfile:
        path = Path(logfile)
        if path.parent and str(path.parent) != ".":
            path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            str(path), maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
    else:
        handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(formatter)
    logger.addHandler(handler)
