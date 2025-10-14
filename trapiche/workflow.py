"""Simple workflow orchestration for Trapiche.

This module implements a lightweight pipeline that runs the two main parts of
the analysis described in the project: text-based biome prediction and
taxonomy-based prediction (community2vec -> TaxonomyToBiome). The functions
are intentionally small and pure where possible to keep the tool maintainable.

The public function is `run_workflow(samples, run_text=True, run_taxonomy=True)`
which accepts a sequence of dicts where each dict has at minimum the keys
`project_description_file_path` (optional) and `taxonomy_files_paths` (list).
The function returns a new list of dicts where each dict is the original one
augmented with analysis results under keys `text_predictions`,
`community_vector` and `taxonomy_prediction` when available.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence, List, Dict
from pathlib import Path
import numpy as np

from . import community2vec as c2v_mod
from . import trapiche_text as tt
from . import deep_pred
from .config import TextToBiomeParams, TaxonomyToBiomeParams


def _read_text_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Text file not found: {path}")
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except UnicodeDecodeError:
        return p.read_text(encoding="ISO-8859-1", errors="replace")


def run_text_step(samples: Sequence[Dict[str, Any]], model_path: Optional[str] = None) -> List[Optional[List[str]]]:
    """Run TextToBiome on the samples, deduplicating identical file paths.

    Returns a list aligned with `samples` where each element is either the
    list of predicted labels or None if no text was provided for that sample.
    """
    # Collect unique text contents (either from inline text or from files)
    # and map samples -> unique index. Inline `project_description_text` has
    # priority over `project_description_file_path`.
    content_to_index: Dict[str, int] = {}
    unique_texts: List[str] = []
    sample_to_unique_idx: List[Optional[int]] = []

    for s in samples:
        inline = s.get("project_description_text")
        if inline is not None:
            # Ensure non-bytes and normalise to str. This is already treated as
            # the file content would be (decoded string).
            text_content = str(inline).encode("utf-8", errors="replace").decode("utf-8")
        else:
            tpath = s.get("project_description_file_path")
            if not tpath:
                sample_to_unique_idx.append(None)
                continue
            # Read from file and use its decoded text
            text_content = _read_text_file(str(tpath))

        # Deduplicate based on the actual text content
        if text_content not in content_to_index:
            content_to_index[text_content] = len(unique_texts)
            unique_texts.append(text_content)
        sample_to_unique_idx.append(content_to_index[text_content])

    if not unique_texts:
        return [None for _ in samples]

    # Change model_path in TextToBiomeParams
    if model_path is not None:
        params_obj = TextToBiomeParams(model_path=model_path)
        preds_unique = tt.predict(
            unique_texts,
            model_path=params_obj.model_path,
            device=params_obj.device,
            max_length=params_obj.max_length,
            threshold_rule=params_obj.threshold_rule,
            split_sentences=params_obj.split_sentences,
        )
    else:
        # Let `trapiche_text.predict` use its own defaults
        preds_unique = tt.predict(unique_texts)

    # Map back to samples
    results: List[Optional[List[str]]] = []
    for idx in sample_to_unique_idx:
        if idx is None:
            results.append(None)
        else:
            results.append(preds_unique[idx])
    return results


def run_vectorise_step(samples: Sequence[Dict[str, Any]]) -> np.ndarray:
    """Run Community2vec.transform over the samples.

    For efficiency this function will deduplicate identical taxonomy file lists
    (order-insensitive) so repeated samples reuse the same vector.
    Returns a list of numpy arrays (one per sample) with the community vector
    or empty arrays if no vector was produced for that sample.
    """
    # Build keys based on sorted tuple of paths so identical sets deduplicate
    sample_lists: List[List[str]] = []

    for s in samples:
        tax_list = s.get("taxonomy_files_paths") or []
        # ensure strings
        sample_lists.append([str(x) for x in tax_list])

    if not sample_lists:
        # no taxonomy files present
        return np.array([np.zeros((0,)) for _ in samples])

    # Call the vectoriser directly (avoids importing the API wrapper)
    return c2v_mod.vectorise_sample(sample_lists)


def run_taxonomy_step(samples: Sequence[Dict[str, Any]], community_vectors: Optional[ np.ndarray | Sequence] = None, text_constraints: Optional[Sequence[Optional[List[str]]]] = None) -> Sequence[Optional[Dict[str, Any]]]:
    """Run TaxonomyToBiome using community vectors and optional text constraints.
    Returns for each sample either a dict (row-wise prediction) or None.
    If community_vectors is None the function computes them.
    """
    if community_vectors is None:
        community_vectors = run_vectorise_step(samples)

    # Constraints: pass text constraints as-is (list per sample) or None
    constrain = [c if c is not None else [] for c in (text_constraints or [None]*len(samples))]

    # Use deep_pred.predict_runs directly and feed it default params from
    # TaxonomyToBiomeParams (this mirrors the behaviour of the API wrapper).
    _params = TaxonomyToBiomeParams()
    df, _ = deep_pred.predict_runs(
        community_vectors=community_vectors,
        return_full_preds=True,
        constrain=constrain,
        batch_size=_params.batch_size,
        k_knn=_params.k_knn,
        dominance_threshold=_params.dominance_threshold,
    )

    # Convert dataframe rows to dicts (one per sample). Ensure we return a
    # Sequence[Optional[Dict[str, Any]]] with the same length as `samples`.
    try:
        if df is None:
            rows = [None] * len(samples)
        else:
            records = df.to_dict(orient="records")
            # Make each record a plain dict[str, Any]
            rows = [({str(k): v for k, v in r.items()} if r is not None else None) for r in records]
            # Pad or truncate to match samples length
            if len(rows) < len(samples):
                rows.extend([None] * (len(samples) - len(rows)))
            elif len(rows) > len(samples):
                rows = rows[: len(samples)]
    except Exception:
        rows = [None] * len(samples)
    return rows


def run_workflow(samples: Sequence[Dict[str, Any]], run_text: bool = True, run_vectorise: bool = True, run_taxonomy: bool = True) -> Sequence[Dict[str, Any]]:
    """Run the requested steps and return augmented sample dicts.

    The input `samples` is not modified; a new list with shallow-copied dicts is
    returned where each dict may include the keys `text_predictions`,
    `community_vector` and `taxonomy_prediction` depending on which steps were run.
    """

    text_results = None
    if run_text:
        text_results = run_text_step(samples)
    else:
        text_results = [None for _ in samples]

    # If taxonomy is requested, but not comm2vec compute community vectors (used internally too)
    if run_vectorise or run_taxonomy:
        community_vectors = run_vectorise_step(samples)
    else:
        community_vectors = [None for _ in samples]

    if run_taxonomy:
        taxonomy_results = run_taxonomy_step(samples, community_vectors=community_vectors, text_constraints=text_results)
    else:
        taxonomy_results = [None for _ in samples]

    # construct output Sequence[Dict[str, Any]]
    result = [d.copy() for d in samples]
    for s, tr, cv, txr in zip(result, text_results, community_vectors, taxonomy_results):
        if tr is not None:
            s["text_predictions"] = tr
        if cv is not None:
            s["community_vector"] = cv.tolist() if cv.size else None
        if txr is not None:
            if txr is not None:
                s.update(txr)  # taxonomy results are a dict with several keys

    return result