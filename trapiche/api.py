import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    TaxonomyToBiomeParams,
    TaxonomyToVectorParams,
    TextToBiomeParams,
    TrapicheWorkflowParams,
    setup_logging,
)
from .workflow import run_workflow

setup_logging(logfile=None)
logger = logging.getLogger(__name__)


class Community2vec:
    """Vectorise taxonomy annotations into community embeddings.

    Use explicit model parameters to resolve assets. Defaults come from
    TaxonomyToVectorParams when not provided.
    """

    def __init__(self, model_name: str | None = None, model_version: str | None = None):
        # If not provided, default to config defaults
        if model_name is None or model_version is None:
            from .config import TaxonomyToVectorParams as _T2V

            _p = _T2V()
            self.model_name = model_name or _p.hf_model
            self.model_version = model_version or _p.model_version
        else:
            self.model_name = model_name
            self.model_version = model_version
        logger.info(
            "Community2vec created",
            extra={"model_name": self.model_name, "model_version": self.model_version},
        )

    def transform(self, samples_sequence: Sequence[dict[str, Any]]) -> np.ndarray:
        """Vectorise one or many samples from taxonomy files.

        Args:
            samples_sequence: Sequence[dict[str,Any]] must have EITHER keys:
              - {"sample_taxonomy_paths": ...}
              - {"study_taxonomy_path": ..., "sample_id": ...}

        Returns:
            np.ndarray: Matrix with shape (n_samples, embedding_dim). If no
            vectors are produced, returns shape (n_samples, 0) or (0, 0) when
            there are no samples.
        """
        from .taxonomy_vectorization import vectorise_samples

        logger.info(
            "Vectorising samples",
            extra={"model_name": self.model_name, "model_version": self.model_version},
        )
        self.vectorised_samples = vectorise_samples(
            samples_sequence, model_name=self.model_name, model_version=self.model_version
        )
        return self.vectorised_samples

    def save(self, path: str | Path) -> None:
        """Save the vectorised samples to a .npy file."""
        if not hasattr(self, "vectorised_samples"):
            raise ValueError("No vectorised samples to save. Call transform() first.")
        logger.info("Saving vectorised samples", extra={"path": str(path)})
        np.save(path, self.vectorised_samples)


class TaxonomyToBiome:
    """Predict biome lineage from community vectors.

    Heavy libraries are imported at prediction time to keep import cost low.
    """

    def __init__(self, params: TaxonomyToBiomeParams | None = None):
        self.params = params or TaxonomyToBiomeParams()
        logger.info(
            "TaxonomyToBiome created",
            extra={
                "params": (
                    self.params.__dict__ if hasattr(self.params, "__dict__") else str(self.params)
                )
            },
        )

    def predict(
        self,
        community_vectors: np.ndarray,
        constrain: Sequence[Sequence[str]] | None = None,
        params: TaxonomyToBiomeParams | None = None,
        *,
        model_name: str | None = None,
        model_version: str | None = None,
    ):
        """Run taxonomy-based prediction.

        Args:
            community_vectors: Array of shape (n_samples, dim).
            constrain: Optional per-sample candidate labels.
            params: Optional override of instance parameters.

        Returns:
            list[dict]: One result dict per sample with prediction keys.
        """
        _params = params or self.params
        # Local import to avoid circular dependency (taxonomy_prediction imports Community2vec from this module)
        from .taxonomy_prediction import predict_runs  # type: ignore

        logger.info(
            "Running taxonomy prediction | community_vectors_shape=%s | params=%s",
            getattr(community_vectors, "shape", None),
            getattr(_params, "__dict__", str(_params)),
        )
        self.results = predict_runs(
            community_vectors=community_vectors, constrain=constrain, params=_params
        )
        return self.results

    def save(self, path: str | Path) -> None:
        """Save the predictions list of dicts to ndjson file."""
        if not hasattr(self, "results"):
            raise ValueError("No predictions to save. Call predict() first.")
        logger.info("Saving taxonomy predictions", extra={"path": str(path)})
        with open(path, "w", encoding="utf-8") as f:
            for record in self.results:
                json.dump(record, f)
                f.write("\n")


class TextToBiome:
    """Predict biome labels from free-form descriptions.

    Parameters are grouped in a dataclass. Heavy imports are deferred until
    prediction time.
    """

    def __init__(self, params: TextToBiomeParams | None = None) -> None:
        self.params = params or TextToBiomeParams()
        self.predictions_: Sequence[dict[str, float] | None] | None = (
            None  # last predictions (optional convenience)
        )
        logger.info(
            "TextToBiome created",
            extra={
                "params": (
                    self.params.__dict__ if hasattr(self.params, "__dict__") else str(self.params)
                )
            },
        )

    def predict(
        self,
        texts: Sequence[str] | str,
        params: TextToBiomeParams | None = None,
    ) -> Sequence[dict[str, float] | None] | None:
        """Run text-based biome prediction.

        Args:
            texts: One or more input texts.
            params: Optional override of instance parameters.

        Returns:
            list[list[str]]: Predicted labels per text.
        """
        # Local import to defer transformers and friends until needed
        from . import text_prediction as tt  # type: ignore

        _p = params or self.params
        logger.info("Running text prediction")
        preds = tt.predict(
            texts,
            model_name=_p.hf_model,
            model_version=_p.model_version,
            device=_p.device,
            max_length=_p.max_length,
            threshold_rule=_p.threshold_rule,
            split_sentences=_p.split_sentences,
        )
        self.predictions_ = preds
        return preds

    def save(self, path: str | Path) -> None:
        """Save the latest predictions to a JSON file.

        Note: This is a convenience method mirroring other wrappers; call
        predict() first to populate predictions_.
        """
        if self.predictions_ is None:
            raise ValueError("No predictions to save. Call predict() first.")
        logger.info("Saving text predictions", extra={"path": str(path)})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.predictions_, f, indent=2)


class TrapicheWorkflowFromSequence:
    """Run the Trapiche workflow over a sequence of samples.

    The wrapper uses in-repo pure functions and respects the provided
    TrapicheWorkflowParams to select steps and outputs.
    """

    def __init__(
        self,
        workflow_params: TrapicheWorkflowParams | None = None,
        text_params: TextToBiomeParams | None = None,
        vectorise_params: TaxonomyToVectorParams | None = None,
        taxonomy_params: TaxonomyToBiomeParams | None = None,
    ) -> None:
        self.workflow_params = workflow_params or TrapicheWorkflowParams()
        self.text_params = text_params or TextToBiomeParams()
        self.vectorise_params = vectorise_params or TaxonomyToVectorParams()
        self.taxonomy_params = taxonomy_params or TaxonomyToBiomeParams()

        logger.info(
            "TrapicheWorkflowFromSequence created",
        )
        logger.info("workflow_params | %s", self.workflow_params)
        logger.info("text_params | %s", self.text_params)
        logger.info("vectorise_params | %s", self.vectorise_params)
        logger.info("taxonomy_params | %s", self.taxonomy_params)

    def run(self, samples: Sequence[dict[str, Any]]) -> Sequence[dict[str, Any]]:
        """Execute the configured steps on samples.

        Args:
            samples: Input sample dicts. Optional keys include
                project_description_file_path and sample_taxonomy_paths.

        Returns:
            list[dict]: Shallow copies of inputs augmented with results.
        """

        result = run_workflow(
            samples,
            run_text=self.workflow_params.run_text,
            run_vectorise=self.workflow_params.run_vectorise,
            run_taxonomy=self.workflow_params.run_taxonomy,
            text_params=self.text_params,
            vectorise_params=self.vectorise_params,
            taxonomy_params=self.taxonomy_params,
            sample_study_text_heuristic=self.workflow_params.sample_study_text_heuristic,
        )

        # Build study-level summary object from the unfiltered results. This
        # keeps original identifiers and predictions intact regardless of
        # output filtering configuration.
        if self.workflow_params.run_study_summary:
            try:
                self.study_summary = self.build_study_summary(
                    result,
                    confidence_threshold=self.workflow_params.study_summary_confidence_threshold,
                )
            except Exception as e:
                logger.warning("Failed to build study summary: %s", e)

        # process which keys to keep according to config
        keep_keys = set()
        sample_keys = set(samples[0].keys()) if samples else set()
        all_keys = set(result[0].keys()) if result else set()
        if self.workflow_params.output_keys:
            keep_keys = set(self.workflow_params.output_keys)
        else:
            keep_keys = set(result[0].keys()) if result else set()
            if not self.workflow_params.keep_text_results:
                keep_keys.discard("text_predictions")
            if not self.workflow_params.keep_vectorise_results:
                keep_keys.discard("community_vector")
            if not self.workflow_params.keep_taxonomy_results:
                # TODO: add a tag to the taxonomy results dict to identify its keys
                taxonomy_keys = all_keys - ({"text_predictions", "community_vector"} | sample_keys)
                keep_keys -= taxonomy_keys

        self.filtered = []
        for r in result:
            newr: dict[str, Any] = {}
            for k in keep_keys:
                if k in r:
                    newr[k] = r[k]
            self.filtered.append(newr)
        logger.info("Workflow finished", extra={"n_results": len(self.filtered)})
        return self.filtered

    def save(self, path: str | Path) -> None:
        """Save the latest filtered results to an NDJSON file.

        Call run() first to populate filtered.
        """
        if not hasattr(self, "filtered"):
            raise ValueError("No results to save. Call run() first.")
        with open(path, "w", encoding="utf-8") as f:
            for record in self.filtered:
                json.dump(record, f)
                f.write("\n")

    def build_study_summary(  # TODO: implement this function to do study-level summaries.
        self,
        results: Sequence[dict[str, Any]],
        *,
        confidence_threshold: float | None = None,
    ) -> dict:
        """Build a study-level biome summary object from per-sample results.

        Groups samples by `project_id` and aggregates the final predictions into
        two mappings per study:
          - confident: Dict[biome_string, List[sample_id]]
          - low_confidence: Dict[biome_string, List[sample_id]]

        Confidence is decided by comparing the selected prediction score
        against `confidence_threshold`. If the threshold is None, the default
        from workflow params is used.

        Args:
            results: Per-sample result dicts (should include project_id,
                sample_id, and a `final_selected_prediction` dict mapping
                lineage to score).
            confidence_threshold: Optional override for confidence threshold.

        Returns:
            dict: Mapping of project_id to a summary object with `confident` and
            `low_confidence` keys.
        """
        th = (
            float(confidence_threshold)
            if confidence_threshold is not None
            else float(self.workflow_params.study_summary_confidence_threshold)
        )

        studies: dict[str, dict[str, dict[str, list[str]]]] = {}

        for rec in results:
            project_id = rec.get("project_id")
            sample_id = rec.get("sample_id")
            if not project_id or not sample_id:
                # Skip records without identifiers
                continue

            final_pred = rec.get("final_selected_prediction")
            if not isinstance(final_pred, dict) or not final_pred:
                # No prediction available; treat as low confidence under a None label bucket
                biome = None
                score = 0.0
            else:
                biome = next(iter(final_pred.keys()))
                score = float(final_pred.get(biome, 0.0))

            bucket = "confident" if score >= th else "low_confidence"
            biome_key = biome if biome is not None else "__unknown__"

            studies.setdefault(project_id, {"confident": {}, "low_confidence": {}})
            studies[project_id][bucket].setdefault(biome_key, []).append(sample_id)

        return studies
