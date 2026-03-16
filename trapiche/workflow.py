"""Simple workflow orchestration for Trapiche.

This module implements a lightweight pipeline that runs the two main parts of
the analysis described in the project: text-based biome prediction and
taxonomy-based prediction (taxonomy_vectorization -> TaxonomyToBiome). The functions
are intentionally small and pure where possible to keep the tool maintainable.

The public function is `run_workflow(samples, run_text=True, run_taxonomy=True)`
which accepts a sequence of dicts where each dict has at minimum the keys
`project_description_file_path` (optional) and `sample_taxonomy_paths` (list).
The function returns a new list of dicts where each dict is the original one
augmented with analysis results under keys `text_predictions`,
`community_vector` and `taxonomy_prediction` when available.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import taxonomy_prediction, taxonomy_vectorization as c2v_mod, text_prediction as tt
from .config import TaxonomyToBiomeParams, TaxonomyToVectorParams, TextToBiomeParams
from .utils import normalize_and_canonicalize_labels, obj_to_serializable

logger = logging.getLogger(__name__)


def _apply_heuristic(
    proj_preds: dict[str, float], samp_preds: dict[str, float]
) -> dict[str, float]:
    """Apply the project/sample label heuristic.

    Keeps sample-level labels that are a substring of (or contain) any
    project-level label. Falls back to project labels plus the top sample
    prediction when no intersection is found.

    Args:
        proj_preds: Project-level label → score dict.
        samp_preds: Sample-level label → score dict.

    Returns:
        dict[str, float]: Combined label → score dict.
    """
    selected = {}
    for sp, _prob in samp_preds.items():
        for pp in proj_preds:
            if sp in pp or pp in sp:
                selected[sp] = _prob
    if not selected:
        selected.update(proj_preds)
        if samp_preds:
            top_samp_pred = max(samp_preds.keys(), key=lambda k: samp_preds.get(k, 0.0))
            selected[top_samp_pred] = float(samp_preds.get(top_samp_pred, 0.0))
    return selected


def _run_text_step_external(
    samples: Sequence[dict[str, Any]],
    use_heuristic: bool,
) -> tuple[
    list[dict[str, float] | None],
    list[dict[str, float] | None],
    list[dict[str, float] | None],
    list[bool],
    list[list[str] | None],
    list[list[str] | None],
]:
    """Build text predictions from external label keys in each sample dict.

    Reads ``ext_text_pred_project`` and ``ext_text_pred_sample`` from each
    sample, validates them, and converts to ``{label: 1.0}`` dicts. When
    ``use_heuristic`` is True and both keys are present for a sample, the
    same ``_apply_heuristic`` logic as the internal pathway is applied.

    Args:
        samples: Sequence of sample dicts.
        use_heuristic: Whether to apply the project/sample heuristic.

    Returns:
        tuple: (combined, project_only, sample_only, heuristic_flags,
        raw_proj_per_sample, raw_samp_per_sample), each aligned to samples.
    """
    combined_results: list[dict[str, float] | None] = []
    proj_preds_per_sample: list[dict[str, float] | None] = []
    samp_preds_per_sample: list[dict[str, float] | None] = []
    heuristic_flags: list[bool] = []
    raw_proj_per_sample: list[list[str] | None] = []
    raw_samp_per_sample: list[list[str] | None] = []

    for position, s in enumerate(samples):
        raw_proj = s.get("ext_text_pred_project")
        raw_samp = s.get("ext_text_pred_sample")

        # Capture raw labels before any canonicalization
        raw_ext_proj: list[str] | None = list(raw_proj) if raw_proj else None
        raw_ext_samp: list[str] | None = list(raw_samp) if raw_samp else None

        # Treat empty list as absent
        proj_labels: list[str] | None = raw_proj if raw_proj else None
        samp_labels: list[str] | None = raw_samp if raw_samp else None

        if proj_labels is None and samp_labels is None:
            logger.warning(
                f"Sample at position {position} has neither ext_text_pred_project "
                "nor ext_text_pred_sample; text predictions will be None."
            )
            combined_results.append(None)
            proj_preds_per_sample.append(None)
            samp_preds_per_sample.append(None)
            heuristic_flags.append(False)
            raw_proj_per_sample.append(None)
            raw_samp_per_sample.append(None)
            continue

        proj_preds: dict[str, float] | None = None
        if proj_labels is not None:
            proj_labels = normalize_and_canonicalize_labels(
                proj_labels,
                warn_prefix=f"Sample pos {position} (project): ",
                fuzzy_fallback=True,
            )
            if not proj_labels:
                proj_labels = None
            else:
                proj_preds = {label: 1.0 for label in proj_labels}

        samp_preds: dict[str, float] | None = None
        if samp_labels is not None:
            samp_labels = normalize_and_canonicalize_labels(
                samp_labels,
                warn_prefix=f"Sample pos {position} (sample): ",
                fuzzy_fallback=True,
            )
            if not samp_labels:
                samp_labels = None
            else:
                samp_preds = {label: 1.0 for label in samp_labels}

        heuristic_active = use_heuristic and proj_preds is not None and samp_preds is not None
        if heuristic_active:
            heuristic_result = _apply_heuristic(proj_preds, samp_preds)
            combined_results.append(heuristic_result if heuristic_result else proj_preds)
            proj_preds_per_sample.append(proj_preds)
            samp_preds_per_sample.append(samp_preds)
            heuristic_flags.append(True)
        else:
            combined = proj_preds if proj_preds is not None else samp_preds
            combined_results.append(combined)
            proj_preds_per_sample.append(proj_preds)
            samp_preds_per_sample.append(samp_preds if use_heuristic else None)
            heuristic_flags.append(False)

        raw_proj_per_sample.append(raw_ext_proj)
        raw_samp_per_sample.append(raw_ext_samp)

    return (
        combined_results,
        proj_preds_per_sample,
        samp_preds_per_sample,
        heuristic_flags,
        raw_proj_per_sample,
        raw_samp_per_sample,
    )


def _read_text_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Text file not found: {path}")
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except UnicodeDecodeError:
        return p.read_text(encoding="ISO-8859-1", errors="replace")


def run_text_step(
    samples: Sequence[dict[str, Any]],
    params_obj: TextToBiomeParams,
    use_heuristic: bool = False,
) -> tuple[
    list[dict[str, float] | None],  # combined predictions (used for constraints)
    list[dict[str, float] | None],  # project text predictions
    list[dict[str, float] | None],  # sample text predictions (when heuristic active)
    list[bool],  # heuristic applied per-sample
    list[list[str] | None],  # raw external project labels (None for internal pathway)
    list[list[str] | None],  # raw external sample labels (None for internal pathway)
]:
    """Predict biome labels from texts with an optional heuristic.

    - When use_heuristic is False, use project text only (inline or from
      project_description_file_path), de-duplicate globally, predict once per
      unique text, and map back to samples.
    - When use_heuristic is True and both project_description_text and
      sample_description_text are present, predict both and take the union of
      labels. If no union is possible, fall back to project labels.

    Returns:
        tuple: (combined, project_only, sample_only, heuristic_flags,
        raw_proj_labels, raw_samp_labels), each aligned to samples.
        raw_proj_labels and raw_samp_labels are None for the internal BERT
        pathway.
    """

    logger.info(f"Running TextToBiome step | use_heuristic={use_heuristic}")

    has_external = any(
        s.get("ext_text_pred_project") or s.get("ext_text_pred_sample") for s in samples
    )
    if has_external:
        logger.info(
            "External text predictions detected — skipping internal model for entire batch."
        )
        return _run_text_step_external(samples, use_heuristic)

    # Helper to normalise text content
    def _norm(v: Any) -> str:
        return str(v).encode("utf-8", errors="replace").decode("utf-8")

    # Collect unique contents for project and sample texts separately to avoid duplicate predictions.
    proj_content_to_idx: dict[str, int] = {}
    proj_unique_texts: list[str] = []
    samp_content_to_idx: dict[str, int] = {}
    samp_unique_texts: list[str] = []

    # Per-sample indices into the unique arrays
    per_sample_proj_idx: list[int | None] = []
    per_sample_samp_idx: list[int | None] = []

    for s in samples:
        # Resolve project text: inline text has priority, else from file path
        proj_inline = s.get("project_description_text")
        proj_text: str | None = None
        if proj_inline is not None:
            proj_text = _norm(proj_inline)
        else:
            proj_path = s.get("project_description_file_path")
            if proj_path:
                proj_text = _read_text_file(str(proj_path))

        # Resolve sample text only used when heuristic is enabled AND project text exists
        # (heuristic applies when both fields are present)
        samp_inline = (
            s.get("sample_description_text")
            if (
                use_heuristic
                and proj_text is not None
                and ("project_description_text" in s)
                and ("sample_description_text" in s)
            )
            else None
        )
        samp_text: str | None = _norm(samp_inline) if samp_inline is not None else None

        # Map project text to unique index
        if proj_text is None:
            per_sample_proj_idx.append(None)
        else:
            if proj_text not in proj_content_to_idx:
                proj_content_to_idx[proj_text] = len(proj_unique_texts)
                proj_unique_texts.append(proj_text)
            per_sample_proj_idx.append(proj_content_to_idx[proj_text])

        # Map sample text to unique index (only when provided and heuristic on)
        if samp_text is None:
            per_sample_samp_idx.append(None)
        else:
            if samp_text not in samp_content_to_idx:
                samp_content_to_idx[samp_text] = len(samp_unique_texts)
                samp_unique_texts.append(samp_text)
            per_sample_samp_idx.append(samp_content_to_idx[samp_text])

    # If there are no texts at all, return all None in all outputs
    if not proj_unique_texts and not samp_unique_texts:
        n = len(samples)
        return (
            [None for _ in samples],
            [None for _ in samples],
            [None for _ in samples],
            [False for _ in range(n)],
            [None for _ in range(n)],
            [None for _ in range(n)],
        )

    # Predict over the union of unique texts to avoid duplicate model calls
    content_to_union_idx: dict[str, int] = {}
    union_unique_texts: list[str] = []
    for t in list(proj_unique_texts) + list(samp_unique_texts):
        if t not in content_to_union_idx:
            content_to_union_idx[t] = len(union_unique_texts)
            union_unique_texts.append(t)

    union_preds_unique: list[dict[str, float]] = []
    if union_unique_texts:
        union_preds_unique = tt.predict(
            union_unique_texts,
            model_name=params_obj.hf_model,
            model_version=params_obj.model_version,
            device=params_obj.device,
            max_length=params_obj.max_length,
            threshold_rule=params_obj.threshold_rule,
            split_sentences=params_obj.split_sentences,
        )

    # Map back to samples combining with heuristic when applicable
    combined_results: list[dict[str, float] | None] = []
    proj_preds_per_sample: list[dict[str, float] | None] = []
    samp_preds_per_sample: list[dict[str, float] | None] = []
    heuristic_flags: list[bool] = []

    for position, (s, pidx, sidx) in enumerate(
        zip(samples, per_sample_proj_idx, per_sample_samp_idx, strict=False)
    ):
        if pidx is None and (sidx is None or not use_heuristic):
            combined_results.append(None)
            proj_preds_per_sample.append(None)
            samp_preds_per_sample.append(None)
            heuristic_flags.append(False)
            continue

        # Project-only prediction via union mapping
        proj_preds = None
        if pidx is not None and proj_unique_texts:
            proj_content = proj_unique_texts[pidx]
            proj_preds = union_preds_unique[content_to_union_idx[proj_content]]

        # Heuristic only when flag is on and both keys exist in the dict
        heuristic_active = (
            use_heuristic and ("project_description_text" in s) and ("sample_description_text" in s)
        )

        # Log messague if heuristic inactive for use_heuristic=True but sample text missing
        if use_heuristic and not heuristic_active:
            logger.info(
                f"Heuristic requested but not applied for sample at position {position}: missing required text fields."
            )

        if heuristic_active and sidx is not None and samp_unique_texts and proj_preds is not None:
            samp_content = samp_unique_texts[sidx]
            samp_preds = union_preds_unique[content_to_union_idx[samp_content]]
            heuristic_result = _apply_heuristic(proj_preds, samp_preds)
            if heuristic_result:
                combined_results.append(heuristic_result)
            else:
                # Fallback to project-level predictions when no intersection
                combined_results.append(proj_preds)
            proj_preds_per_sample.append(proj_preds)
            samp_preds_per_sample.append(samp_preds)
            heuristic_flags.append(True)
        else:
            # Heuristic not active or no sample text: use project-only
            combined = proj_preds
            combined_results.append(combined)
            proj_preds_per_sample.append(proj_preds)
            samp_preds_per_sample.append(None)
            heuristic_flags.append(False)

    return (
        combined_results,
        proj_preds_per_sample,
        samp_preds_per_sample,
        heuristic_flags,
        [None] * len(samples),
        [None] * len(samples),
    )


def run_taxonomy_step(
    samples: Sequence[dict[str, Any]],
    *,
    params: TaxonomyToBiomeParams,
    community_vectors: np.ndarray | Sequence | None = None,
    text_constraints: Sequence[dict[str, float] | None] | None = None,
) -> Sequence[dict[str, Any] | None]:
    """Run TaxonomyToBiome using community vectors and optional text constraints.
    Returns for each sample either a dict (row-wise prediction) or None.
    If community_vectors is None the function computes them.
    """

    logger.info(
        f"Running TaxonomyToBiome step | model_name={params.hf_model} | model_version={params.model_version}"
    )
    if community_vectors is None:
        # When used from run_workflow, we pass community vectors explicitly; keep signature for compatibility.
        _vec_p = TaxonomyToVectorParams()
        community_vectors = c2v_mod.vectorise_samples(
            samples, model_name=_vec_p.hf_model, model_version=_vec_p.model_version
        )

    # Constraints: pass text constraints as-is (list per sample) or None
    constrain = text_constraints or [None] * len(samples)

    # Use taxonomy_prediction.predict_runs directly and feed it default params from
    # TaxonomyToBiomeParams (this mirrors the behaviour of the API wrapper).
    results = taxonomy_prediction.predict_runs(
        community_vectors=community_vectors,
        constrain=constrain,
        params=params,
    )

    return results


def run_workflow(
    samples: Sequence[dict[str, Any]],
    *,
    text_params: TextToBiomeParams,
    vectorise_params: TaxonomyToVectorParams,
    taxonomy_params: TaxonomyToBiomeParams,
    run_text: bool = True,
    run_vectorise: bool = True,
    run_taxonomy: bool = True,
    sample_study_text_heuristic: bool = False,
) -> Sequence[dict[str, Any]]:
    """Run the requested steps and return augmented sample dicts.

    The input `samples` is not modified; a new list with shallow-copied dicts is
    returned where each dict may include the keys `text_predictions`,
    `community_vector` and `taxonomy_prediction` depending on which steps were run.
    """

    logger.info(
        f"Running Trapiche workflow | run_text={run_text} | run_vectorise={run_vectorise} | run_taxonomy={run_taxonomy} | sample_study_text_heuristic={sample_study_text_heuristic}"
    )

    text_results = None
    proj_text_results = None
    samp_text_results = None
    heuristic_flags = None
    raw_proj_text: list[list[str] | None] = [None for _ in samples]
    raw_samp_text: list[list[str] | None] = [None for _ in samples]
    if run_text:
        (
            text_results,
            proj_text_results,
            samp_text_results,
            heuristic_flags,
            raw_proj_text,
            raw_samp_text,
        ) = run_text_step(
            samples, params_obj=text_params, use_heuristic=sample_study_text_heuristic
        )
    else:
        text_results = [None for _ in samples]
        proj_text_results = [None for _ in samples]
        samp_text_results = [None for _ in samples]
        heuristic_flags = [False for _ in samples]

    # If taxonomy is requested, but not taxonomy_vectorization compute community vectors (used internally too)
    if run_vectorise or run_taxonomy:
        community_vectors = c2v_mod.vectorise_samples(
            samples,
            model_name=vectorise_params.hf_model,
            model_version=vectorise_params.model_version,
        )
    else:
        community_vectors = [None for _ in samples]

    if run_taxonomy:
        taxonomy_results = run_taxonomy_step(
            samples,
            params=taxonomy_params,
            community_vectors=community_vectors,
            text_constraints=text_results,
        )
    else:
        taxonomy_results = [None for _ in samples]

    # construct output Sequence[Dict[str, Any]]
    result = [d.copy() for d in samples]
    serializable_results = []
    for s, tr, cv, txr, pr_tr, sa_tr, hflag, raw_proj, raw_samp in zip(
        result,
        text_results,
        community_vectors,
        taxonomy_results,
        proj_text_results,
        samp_text_results,
        heuristic_flags,
        raw_proj_text,
        raw_samp_text,
        strict=True,
    ):
        if tr is not None:
            s["text_predictions"] = tr
        # If heuristic was applied for this sample, also include project and sample text predictions
        if hflag:
            if pr_tr is not None:
                s["text_predictions_project"] = pr_tr
            if sa_tr is not None:
                s["text_predictions_sample"] = sa_tr
        if raw_proj is not None:
            s["_raw_ext_text_pred_project"] = raw_proj
        if raw_samp is not None:
            s["_raw_ext_text_pred_sample"] = raw_samp
        if cv is not None and isinstance(cv, np.ndarray) and np.any(cv):
            s["community_vector"] = cv.tolist() if cv.size else None
            if txr is not None:
                s.update(txr)
        _s = obj_to_serializable(s)  # convert any non-serializable types in-place
        serializable_results.append(_s)

    return serializable_results
