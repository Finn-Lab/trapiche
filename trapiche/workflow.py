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
from .utils import obj_to_serializable


def _read_text_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Text file not found: {path}")
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except UnicodeDecodeError:
        return p.read_text(encoding="ISO-8859-1", errors="replace")


def run_text_step(
    samples: Sequence[Dict[str, Any]],
    params_obj: TextToBiomeParams,
    use_heuristic: bool = False,
) -> tuple[
    List[Optional[List[str]]],  # combined predictions (used for constraints)
    List[Optional[List[str]]],  # project text predictions
    List[Optional[List[str]]],  # sample text predictions (when heuristic active)
    List[bool],                 # heuristic applied per-sample
]:
    """Run TextToBiome on the samples with optional sample-over-study heuristic.

    Behaviour
    - If use_heuristic is False (default):
      Uses either inline 'project_description_text' or a file path at
      'project_description_file_path' for each sample, deduplicates identical
      text contents across samples, runs the text predictor once per unique
      text, and maps the predictions back to each sample.
    - If the heuristic is True and a sample provides both keys
      'project_description_text' and 'sample_description_text', we run the
      predictor for both texts (with global de-duplication) and then intersect
      the two prediction lists, keeping the longest string when one matches the
      other by prefix. If the intersection is empty, we fall back to the
      project-level predictions for that sample.

    Returns a list aligned with `samples` where each element is either the
    list of predicted labels or None if no applicable text was provided.
    """
    # Helper to normalise text content
    def _norm(v: Any) -> str:
        return str(v).encode("utf-8", errors="replace").decode("utf-8")

    # Collect unique contents for project and sample texts separately to avoid duplicate predictions.
    proj_content_to_idx: Dict[str, int] = {}
    proj_unique_texts: List[str] = []
    samp_content_to_idx: Dict[str, int] = {}
    samp_unique_texts: List[str] = []

    # Per-sample indices into the unique arrays
    per_sample_proj_idx: List[Optional[int]] = []
    per_sample_samp_idx: List[Optional[int]] = []

    for s in samples:
        # Resolve project text: inline text has priority, else from file path
        proj_inline = s.get("project_description_text")
        proj_text: Optional[str] = None
        if proj_inline is not None:
            proj_text = _norm(proj_inline)
        else:
            proj_path = s.get("project_description_file_path")
            if proj_path:
                proj_text = _read_text_file(str(proj_path))
        
        # Resolve sample text only used when heuristic is enabled AND project text exists
        # (heuristic applies when both fields are present)
        samp_inline = s.get("sample_description_text") if (
            use_heuristic
            and proj_text is not None
            and ("project_description_text" in s)
            and ("sample_description_text" in s)
        ) else None
        samp_text: Optional[str] = _norm(samp_inline) if samp_inline is not None else None

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
        return [None for _ in samples], [None for _ in samples], [None for _ in samples], [False for _ in range(n)]

    # Predict over the union of unique texts to avoid duplicate model calls
    content_to_union_idx: Dict[str, int] = {}
    union_unique_texts: List[str] = []
    for t in list(proj_unique_texts) + list(samp_unique_texts):
        if t not in content_to_union_idx:
            content_to_union_idx[t] = len(union_unique_texts)
            union_unique_texts.append(t)

    union_preds_unique: List[List[str]] = []
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

    # Helper for heuristic intersection keeping the longest prefix match
    def intersect_keep_longest(a: List[str], b: List[str]) -> List[str]:
        selected: set[str] = set()
        for sa in a:
            for sb in b:
                if sa.startswith(sb) or sb.startswith(sa):
                    selected.add(sa if len(sa) >= len(sb) else sb)
        return list(selected)

    # Map back to samples combining with heuristic when applicable
    combined_results: List[Optional[List[str]]] = []
    proj_preds_per_sample: List[Optional[List[str]]] = []
    samp_preds_per_sample: List[Optional[List[str]]] = []
    heuristic_flags: List[bool] = []

    for s, pidx, sidx in zip(samples, per_sample_proj_idx, per_sample_samp_idx):
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
        heuristic_active = use_heuristic and ("project_description_text" in s) and ("sample_description_text" in s)

        if heuristic_active and sidx is not None and samp_unique_texts and proj_preds is not None:
            samp_content = samp_unique_texts[sidx]
            samp_preds = union_preds_unique[content_to_union_idx[samp_content]]
            inter = intersect_keep_longest(proj_preds, samp_preds)
            if inter:
                combined_results.append(sorted(inter))
            else:
                # Fallback to project-level predictions when no intersection
                combined_results.append(list(proj_preds))
            proj_preds_per_sample.append(list(proj_preds) if proj_preds is not None else None)
            samp_preds_per_sample.append(list(samp_preds) if samp_preds is not None else None)
            heuristic_flags.append(True)
        else:
            # Heuristic not active or no sample text: use project-only
            combined = list(proj_preds) if proj_preds is not None else None
            combined_results.append(combined)
            proj_preds_per_sample.append(list(proj_preds) if proj_preds is not None else None)
            samp_preds_per_sample.append(None)
            heuristic_flags.append(False)

    return combined_results, proj_preds_per_sample, samp_preds_per_sample, heuristic_flags


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

def run_workflow(samples: Sequence[Dict[str, Any]], *,text_params: TextToBiomeParams,  vectorise_params: TaxonomyToVectorParams, taxonomy_params: TaxonomyToBiomeParams, run_text: bool = True, run_vectorise: bool = True, run_taxonomy: bool = True, sample_over_study_heuristic: bool = False) -> Sequence[Dict[str, Any]]:
    """Run the requested steps and return augmented sample dicts.

    The input `samples` is not modified; a new list with shallow-copied dicts is
    returned where each dict may include the keys `text_predictions`,
    `community_vector` and `taxonomy_prediction` depending on which steps were run.
    """

    text_results = None
    proj_text_results = None
    samp_text_results = None
    heuristic_flags = None
    if run_text:
        text_results, proj_text_results, samp_text_results, heuristic_flags = run_text_step(
            samples, params_obj=text_params, use_heuristic=sample_over_study_heuristic
        )
    else:
        text_results = [None for _ in samples]
        proj_text_results = [None for _ in samples]
        samp_text_results = [None for _ in samples]
        heuristic_flags = [False for _ in samples]

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
    serializable_results = []
    for s, tr, cv, txr, pr_tr, sa_tr, hflag in zip(result, text_results, community_vectors, taxonomy_results, proj_text_results, samp_text_results, heuristic_flags):
        if tr is not None:
            s["text_predictions"] = tr
        # If heuristic was applied for this sample, also include project and sample text predictions
        if hflag:
            if pr_tr is not None:
                s["text_predictions_project"] = pr_tr
            if sa_tr is not None:
                s["text_predictions_sample"] = sa_tr
        if cv is not None and isinstance(cv, np.ndarray) and np.any(cv):
            s["community_vector"] = cv.tolist() if cv.size else None
            if txr is not None:
                s.update(txr)  
        _s = obj_to_serializable(s)  # convert any non-serializable types in-place
        serializable_results.append(_s)

    return serializable_results