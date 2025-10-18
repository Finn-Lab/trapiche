"""Simple workflow orchestration for Trapiche.

This module implements a lightweight pipeline that runs the two main parts of
the analysis described in the project: text-based biome prediction and
taxonomy-based prediction (taxonomy_vectorization -> TaxonomyToBiome). The functions
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

from . import taxonomy_vectorization as c2v_mod
from . import text_prediction as tt
from . import taxonomy_prediction
from .config import TaxonomyToVectorParams, TextToBiomeParams, TaxonomyToBiomeParams


def _read_text_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Text file not found: {path}")
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except UnicodeDecodeError:
        return p.read_text(encoding="ISO-8859-1", errors="replace")


def run_text_step(samples: Sequence[Dict[str, Any]], params_obj: TextToBiomeParams) -> List[Optional[List[str]]]:
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

    # Build params for text model
    preds_unique = tt.predict(
        unique_texts,
        model_name=params_obj.hf_model,
        model_version=params_obj.model_version,
        device=params_obj.device,
        max_length=params_obj.max_length,
        threshold_rule=params_obj.threshold_rule,
        split_sentences=params_obj.split_sentences,
    )

    # Map back to samples
    results: List[Optional[List[str]]] = []
    for idx in sample_to_unique_idx:
        if idx is None:
            results.append(None)
        else:
            results.append(preds_unique[idx])
    return results


def run_vectorise_step(samples: Sequence[Dict[str, Any]], *, model_name: str | None = None, model_version: str | None = None) -> np.ndarray:
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
    # Require explicit model parameters
    return c2v_mod.vectorise_sample(sample_lists, model_name=model_name, model_version=model_version)

def run_taxonomy_step(samples: Sequence[Dict[str, Any]], *,params:TaxonomyToBiomeParams, community_vectors: Optional[ np.ndarray | Sequence] = None, text_constraints: Optional[Sequence[Optional[List[str]]]] = None) -> Sequence[Optional[Dict[str, Any]]]:
    """Run TaxonomyToBiome using community vectors and optional text constraints.
    Returns for each sample either a dict (row-wise prediction) or None.
    If community_vectors is None the function computes them.
    """
    if community_vectors is None:
        # When used from run_workflow, we pass community vectors explicitly; keep signature for compatibility.
        community_vectors = run_vectorise_step(samples)  

    # Constraints: pass text constraints as-is (list per sample) or None
    constrain = text_constraints or [None]*len(samples)

    # Use taxonomy_prediction.predict_runs directly and feed it default params from
    # TaxonomyToBiomeParams (this mirrors the behaviour of the API wrapper).
    results = taxonomy_prediction.predict_runs(
        community_vectors=community_vectors,
        constrain=constrain,
        params= params,
    )

    return results

def run_workflow(samples: Sequence[Dict[str, Any]], *,text_params: TextToBiomeParams,  vectorise_params: TaxonomyToVectorParams, taxonomy_params: TaxonomyToBiomeParams, run_text: bool = True, run_vectorise: bool = True, run_taxonomy: bool = True) -> Sequence[Dict[str, Any]]:
    """Run the requested steps and return augmented sample dicts.

    The input `samples` is not modified; a new list with shallow-copied dicts is
    returned where each dict may include the keys `text_predictions`,
    `community_vector` and `taxonomy_prediction` depending on which steps were run.
    """

    text_results = None
    if run_text:
        text_results = run_text_step(samples, params_obj=text_params)
    else:
        text_results = [None for _ in samples]

    # If taxonomy is requested, but not taxonomy_vectorization compute community vectors (used internally too)
    if run_vectorise or run_taxonomy:
        community_vectors = run_vectorise_step(samples, model_name=vectorise_params.hf_model, model_version=vectorise_params.model_version)
    else:
        community_vectors = [None for _ in samples]

    if run_taxonomy:
        taxonomy_results = run_taxonomy_step(samples, params=taxonomy_params, community_vectors=community_vectors, text_constraints=text_results)
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
            s.update(txr)  # taxonomy results are a dict with several keys

    return result