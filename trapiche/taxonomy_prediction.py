"""Deep learning based taxonomy prediction pipeline.

Includes model loading (lazy), vectorisation, consensus heuristics and
prediction refinement utilities.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
from collections import defaultdict
from collections.abc import Sequence
from contextlib import suppress
from functools import lru_cache
from itertools import combinations
from statistics import harmonic_mean
from typing import Any, cast

import numpy as np
from more_itertools import chunked

from .config import TaxonomyToBiomeParams, TaxonomyToVectorParams
from .taxonomy_vectorization import load_mgnify_c2v
from .utils import (
    _get_hf_model_path,
    cosine_similarity_pairwise,
    get_similar_predictions,
    load_biome_herarchy_dict,
    shared_asset_params,
)

logger = logging.getLogger(__name__)

KNN_VALUE_DEFAULT = -1  # THE FUNCTION IS DEPRECATED. TODO: Transform function to KNN tool


@lru_cache
def load_biome_tags_list(
    model_name: str | None = None,
    model_version: str | None = None,
    local_model_dir: str | None = None,
):
    """Load tag list for the taxonomy classifier from the vectorizer repo assets.

    Args:
        model_name: HF repository id. Defaults to TaxonomyToVectorParams.hf_model.
        model_version: Model version. Defaults to TaxonomyToVectorParams.model_version.
        local_model_dir: Optional local directory to resolve the asset from
            instead of Hugging Face Hub. Defaults to
            TaxonomyToVectorParams.local_model_dir when not provided.

    Returns:
        list[str]: Flat list of tag strings.
    """
    if model_name is None or model_version is None:
        _name, _version, _dir = shared_asset_params()
        model_name = model_name or _name
        model_version = model_version or _version
        if local_model_dir is None:
            local_model_dir = _dir
    tags_dct_file = _get_hf_model_path(
        model_name, model_version, "biome_tags_*.json", local_model_dir
    )
    logger.debug(f"Loading biome tags dictionary from file={tags_dct_file}")
    with open(tags_dct_file) as h:
        tags_dct = json.load(h)
    tags_li = list(tags_dct)
    return tags_li


def generate_all_combinations(s: Sequence[str]):
    """Generate all combinations (powerset) of a sequence.

    Args:
        s: Input sequence of strings.

    Returns:
        list[tuple[str, ...]]: All combinations including empty tuple.
    """
    result = []
    for r in range(len(s) + 1):
        result.extend(combinations(s, r))
    return result


@lru_cache
def load_tag_biomes(
    model_name: str | None = None,
    model_version: str | None = None,
    local_model_dir: str | None = None,
):
    """Map tag combinations to canonical biome lineage.

    Args:
        model_name: Vectorizer HF repository id (defaults to config).
        model_version: Vectorizer model version (defaults to config).
        local_model_dir: Optional local directory for offline resolution.

    Returns:
        tuple[dict[str, str], dict[str, tuple[set[str], int]]]:
        A mapping from tag combination to lineage, and auxiliary metadata.
    """
    biome_herarchy_dct, _ = load_biome_herarchy_dict(model_name, model_version, local_model_dir)
    tags_li = load_biome_tags_list(model_name, model_version, local_model_dir)
    bioms = {x: ((set(x.split(":"))), len(x.split(":"))) for x in biome_herarchy_dct.values()}
    tag_biomes = {}
    for _prediction in tags_li:
        _n_pots = _prediction.split("|")
        for comb in generate_all_combinations(_n_pots):
            comb = set(comb)
            sels = [(k, size) for k, (se, size) in bioms.items() if len(comb) == len(comb & se)]
            if not sels:
                _comb = comb - set("Soil|Terrestrial|Non-Defined".split("|"))
                sels = [
                    (k, size) for k, (se, size) in bioms.items() if len(_comb) == len(_comb & se)
                ]
                if not sels:
                    _comb = _comb - {"Rhizosphere"}
                    sels = [
                        (k, size)
                        for k, (se, size) in bioms.items()
                        if len(_comb) == len(_comb & se)
                    ]
            if not sels:
                continue
            sel = sorted(sels, key=lambda x: x[1])[0][0]
            tag_biomes["|".join(sorted(comb))] = sel
    return tag_biomes, bioms


def focal_loss_fixed(y_true, y_pred):
    """Placeholder for focal loss used by legacy model files.

    This function exists to satisfy custom_objects during model load.
    """
    pass


# Lazy TensorFlow import helper and model accessors


def _positive_int_env(name: str) -> int | None:
    """Read an optional positive-integer environment variable.

    Args:
        name: Environment variable name.

    Returns:
        int | None: Parsed value, or None when the variable is unset/empty.

    Raises:
        ValueError: If the value is not a positive integer.
    """
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be a positive integer (got {raw!r})") from e
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero (got {raw!r})")
    return value


def _get_tensorflow():
    """Lazily import TensorFlow with a clear error if unavailable.

    Optional resource limits are read from the environment on every call
    (TensorFlow ignores them after its runtime has been initialised):

    - ``TRAPICHE_TF_GPU_MEMORY_LIMIT_MB``: cap GPU memory per device (MB);
      when unset, GPU memory growth is enabled instead.
    - ``TRAPICHE_TF_NUM_THREADS``: pin intra- and inter-op parallelism to this
      many threads; when unset TensorFlow's defaults (all cores) are kept.
    """
    try:
        tf = importlib.import_module("tensorflow")
    except Exception as e:
        raise RuntimeError(
            "TensorFlow is required for deep prediction but could not be imported. "
            "Install TensorFlow (for CPU-only environments: pip install tensorflow) and ensure it matches your Python version. "
            f"Original error: {e}"
        ) from e

    gpu_limit_mb = _positive_int_env("TRAPICHE_TF_GPU_MEMORY_LIMIT_MB")
    num_threads = _positive_int_env("TRAPICHE_TF_NUM_THREADS")

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        with suppress(RuntimeError):
            if gpu_limit_mb is not None:
                config = [
                    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_limit_mb)
                ]
                tf.config.experimental.set_virtual_device_configuration(gpu, config)
            else:
                tf.config.experimental.set_memory_growth(gpu, True)
    if num_threads is not None:
        with suppress(RuntimeError):
            tf.config.threading.set_intra_op_parallelism_threads(num_threads)
            tf.config.threading.set_inter_op_parallelism_threads(num_threads)
    return tf


# Load the model, including the custom loss function
@lru_cache
def load_custom_model(model_file: str | None = None):
    """Load the Keras model on demand.

    Args:
        model_file: Optional explicit path to the ``.model.h5`` file. When
            given, it is loaded directly and no Hugging Face/local-dir
            resolution is performed. When None (the default), the file is
            resolved via :func:`_resolve_model_file` from a freshly constructed
            `TaxonomyToBiomeParams` (env vars / config file), honoring
            `hf_model`, `model_version`, and `local_model_dir`
            (`TRAPICHE_TAXONOMY_*`). Both call styles share one cache entry
            per resolved path.

    Returns:
        Any: Compiled TensorFlow Keras model instance.

    Raises:
        RuntimeError: If loading fails or TensorFlow is unavailable.
    """
    if model_file is None:
        # Delegate to the explicit-path branch so the cache is keyed by path only.
        return load_custom_model(_resolve_model_file(TaxonomyToBiomeParams()))
    model_path = model_file
    logger.debug(f"Loading model from file={model_path}")

    tf = _get_tensorflow()
    try:
        model = tf.keras.models.load_model(
            model_path, custom_objects={"focal_loss_fixed": focal_loss_fixed}, compile=False
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="categorical_crossentropy",
            metrics=[tf.keras.metrics.AUC()],
        )
        return model
    except Exception as e:
        raise RuntimeError(
            "Failed to load the TensorFlow model. Ensure the model file is compatible with your installed TensorFlow/Keras version "
            f"and that custom objects are provided. File: {model_path}. Original error: {e}"
        ) from e


def bnn_model2gg(*args, **kwargs):
    """Backwards-compatible callable that behaves like the lazily loaded model."""
    return load_custom_model()(*args, **kwargs)


@lru_cache
def _resolve_model_file_cached(
    hf_model: str, model_version: str, local_model_dir: str | None
) -> str:
    return str(
        _get_hf_model_path(
            hf_model, model_version, "taxonomy_to_biome_v*.model.h5", local_model_dir
        )
    )


def _resolve_model_file(params: TaxonomyToBiomeParams) -> str:
    """Resolve the taxonomy classifier asset path for explicit parameters.

    The result is cached per ``(hf_model, model_version, local_model_dir)`` so
    that per-chunk calls do not repeat the Hugging Face Hub lookup.
    """
    return _resolve_model_file_cached(params.hf_model, params.model_version, params.local_model_dir)


def find_best_path(_prediction: str, *, vector_params: TaxonomyToVectorParams | None = None):
    _, bioms = load_tag_biomes(*shared_asset_params(vector_params))
    _n_pots = set(_prediction.split("|"))
    sels = [(k, size) for k, (se, size) in bioms.items() if len(_n_pots) == len(_n_pots & se)]
    sel = sorted(sels, key=lambda x: x[1])[0][0]
    return sel


def from_probs_to_pred(
    _probs,
    potential_space: list[dict[str, float] | None],
    params: TaxonomyToBiomeParams,
    vector_params: TaxonomyToVectorParams | None = None,
) -> tuple[list[dict[str, float] | None], list[dict[str, float] | None]]:
    """Convert class probabilities into top predictions.

    Optionally constrain candidates by matching lineage prefixes from text.

    Args:
        _probs: Array of shape (n_samples, n_classes).
        potential_space: Per-sample iterable of prefix constraints or None.
        params: Thresholds and control parameters.
        vector_params: Optional vectorizer params used to resolve the shared
            biome hierarchy / tag-list assets (offline use).

    Returns:
        tuple: (top_predictions, constrained_top_predictions), both lists of
        dicts mapping lineage to score, aligned to input samples.
    """
    _asset_args = shared_asset_params(vector_params)
    tag_biomes, _ = load_tag_biomes(*_asset_args)
    tags_li = load_biome_tags_list(*_asset_args)

    top_predictions = []
    constrained_top_predictions = []

    #### WORK IN PROGRESS, HERE GET THE TOP PREDS, AND USE THAT TO REFINEMENT. USE get_similar_predictions
    for pr, _pot_space in zip(_probs, potential_space, strict=False):
        if np.isnan(pr).any():
            top_p = None
            constrained_top_p = None
        else:
            # find top predictions, more than one is there is not a lot of certainty
            top_predictions_idx = get_similar_predictions(
                pr, params.top_prob_diff_threshold, params.top_prob_ratio_threshold
            )
            top_p = {
                tag_biomes.get(tags_li[idx], tags_li[idx]): pr[idx] for idx in top_predictions_idx
            }

            if _pot_space is None or len(_pot_space) == 0:
                constrained_top_p = None
            else:
                # Find matching tags in the potential (text) space
                potential_tags = {}
                for k, v in tag_biomes.items():
                    for pot, _prob in _pot_space.items():
                        if re.search("^" + re.escape(pot.strip()), v):
                            potential_tags[k] = _prob

                # If nothing matches, respect the constraint by returning it
                if len(potential_tags) == 0:
                    constrained_top_p = dict(_pot_space)
                else:
                    non_useful_tags = [
                        ix for ix, x in enumerate(tags_li) if x not in potential_tags
                    ]
                    _pr = pr.copy()
                    _pr[non_useful_tags] = -1
                    # Use the masked probability array when computing constrained top predictions
                    constrained_top_predictions_idx = get_similar_predictions(
                        _pr,
                        diff_thresh=params.top_prob_diff_threshold,
                        ratio_thresh=params.top_prob_ratio_threshold,
                    )
                    constrained_top_p = {
                        tag_biomes.get(tags_li[idx], tags_li[idx]): _pr[idx]
                        for idx in constrained_top_predictions_idx
                    }

        top_predictions.append(top_p)
        constrained_top_predictions.append(constrained_top_p)

    return top_predictions, constrained_top_predictions


def get_unanbigious_prediction(co, dominance_threshold, best_lineage_min_depth=4):
    """Compute node frequencies across KNN lineages.

    Args:
        co: Pandas Series with counts by lineage string.
        dominance_threshold: Minimum frequency to keep a node.

    Returns:
        tuple: (node_frequencies, sorted_passed, top_dominant).
    """
    _node_frquencies = {}
    for lineage, count in co.items():
        spl = lineage.split(":")
        for ix in range(1, len(spl) + 1):
            node = ":".join(spl[:ix])
            _node_frquencies.setdefault(node, []).append(count)

    node_frequencies = {k: sum(v) for k, v in _node_frquencies.items()}
    _filtered = [
        (k, v)
        for k, v in node_frequencies.items()
        if v > dominance_threshold
        and k != ""
        and len(k.split(":")) >= best_lineage_min_depth  # Only consider nodes with sufficient depth
    ]
    sorted_passed = sorted(_filtered, key=lambda x: len(x[0].split(":")), reverse=True)
    top_dominant = sorted_passed[0] if sorted_passed else [co.index[0], co.iloc[0]]

    # Claculate score based on mean of score for each lineage containing the top dominant node
    top_dominant_score = None
    if top_dominant is not None:
        top_dominant_score = np.sum([v for k, v in co.items() if top_dominant[0] in k])
        top_dominant = {top_dominant[0]: top_dominant_score}

    return node_frequencies, sorted_passed, top_dominant


def knn_batch(
    predictions: list[list[Any] | None],
    query_vectors: np.ndarray,
    params: TaxonomyToBiomeParams,
    vector_params: TaxonomyToVectorParams | None = None,
) -> list[list[dict[str, Any]] | None]:
    """Find KNN  using cosine similarity in the vector space. Return similar samples, but maximum `max_per_study` per project.



    Parameters
    ----------
    predictions : list of str or None
        List of predicted lineage prefixes from the deep model. Must have the
        same length as query_vectors.
    query_vectors : np.ndarray
        Query embeddings of shape (n_queries, dim).
    params : TaxonomyToBiomeParams
        Contains model names, refinement settings, and thresholds.

    Returns
    -------
    List[Optional[List[Dict[str, Any]]]]
    """

    logger.debug("DEPRECATED: Starting batch KNN refinement of predictions")

    # Prepare result list in input order
    results: list[list[dict[str, Any]] | None] = [None] * len(predictions)
    if KNN_VALUE_DEFAULT <= 0:
        return results

    # Load MGnify sample vectors and metadata
    _hier_name, _hier_version, _hier_dir = shared_asset_params(vector_params)
    mgnify_sample_vectors, mgnify_meta = load_mgnify_c2v(
        model_name=params.hf_model,
        model_version=params.model_version,
        local_model_dir=params.local_model_dir,
        hierarchy_model_name=_hier_name,
        hierarchy_model_version=_hier_version,
        hierarchy_local_model_dir=_hier_dir,
    )

    max_per_study = max(1, KNN_VALUE_DEFAULT // 3)

    # Normalize predictions: None -> empty string

    # Group query indices by unique prediction key
    groups: dict[str, list[int]] = defaultdict(list)
    for ix, pred in enumerate(predictions):
        key = "|".join(sorted(pred)) if pred else None
        if key:
            groups[key].append(ix)

    # Process each group once
    for pred_key, indices in groups.items():
        prediction = pred_key if pred_key else ""

        # Restrict subjects to those matching the predicted prefix
        _subject_df = mgnify_meta[mgnify_meta["BIOME_AMEND"].str.contains(prediction)]
        if _subject_df.empty:
            # Nothing to refine; skip
            continue

        # Select the corresponding subject vectors
        subject_vector = mgnify_sample_vectors.loc[_subject_df.index]

        # Extract relevant query vectors
        query_subset = query_vectors[indices]

        # Compute cosine similarities
        sims = cosine_similarity_pairwise(query_subset, subject_vector)
        sims[np.isnan(sims)] = 0
        # Process each query in this group
        for local_ix in range(len(indices)):
            global_ix = indices[local_ix]

            # Take similarity scores for this query
            sim_scores = sims[local_ix]

            # Sort indices by descending similarity
            sorted_ix = np.argsort(sim_scores)[::-1]

            # Subset dataframe with similarity order
            top_df = _subject_df.iloc[sorted_ix].copy()
            top_df["COSINE_SIMILARITY"] = sim_scores[sorted_ix]

            top_df_limited = top_df.groupby("project_id", group_keys=False).head(max_per_study)

            # Now take the top k_knn overall, after applying the per-study cap
            _selected_df = top_df_limited.head(KNN_VALUE_DEFAULT)

            _selected_df = _selected_df[["ACCESSION", "COSINE_SIMILARITY", "BIOME_AMEND"]]
            _selected_df.columns = ["ACCESSION", "COSINE_SIMILARITY", "BIOME"]

            results[global_ix] = cast(list[dict[str, Any]], _selected_df.to_dict(orient="records"))

            # results[global_ix] = _selected_df.to_dict(orient='records')

    return results


np.seterr(divide="ignore", invalid="ignore")  # handle bad files == divition by zero error


def full_stack_prediction(
    query_vector,
    constrains,
    params: TaxonomyToBiomeParams,
    vector_params: TaxonomyToVectorParams | None = None,
) -> list[dict[str, Any]]:
    """Predict biome using deep model and KNN refinement.

    Applies optional constraints from text and returns per-sample dicts with
    raw, constrained, refined, and final selections. ``vector_params`` is
    forwarded to the shared-asset loaders (biome hierarchy, tag list, KNN
    reference vectors) so explicit ``local_model_dir`` overrides apply.
    """
    # prediction baded on deep learning model
    logger.debug("Starting full stack prediction")
    model = load_custom_model(_resolve_model_file(params))
    deep_l_probs = model(query_vector).numpy()

    top_predictions, constrained_top_predictions = from_probs_to_pred(
        deep_l_probs, potential_space=constrains, params=params, vector_params=vector_params
    )

    # Get unambiguous predictions
    unambiguous_predictions = []
    unambiguous_constrained_predictions = []
    for _top_pred, _constrained_top_pred, constrain in zip(
        top_predictions, constrained_top_predictions, constrains, strict=False
    ):
        # _, _, top_dominant = get_unanbigious_prediction(pd.Series(_top_pred), dominance_threshold=params.dominance_threshold)
        if _top_pred:
            top_dominant_term = max(_top_pred.items(), key=lambda kv: kv[1])[0]
            top_dominant = {top_dominant_term: _top_pred.get(top_dominant_term, 0)}
        else:
            top_dominant = None
        unambiguous_predictions.append(top_dominant)

        if not _constrained_top_pred:
            unambiguous_constrained_predictions.append(None)
            continue

        # _, _, _top_dominant_const = get_unanbigious_prediction(pd.Series(_constrained_top_pred), dominance_threshold=params.dominance_threshold)
        if _constrained_top_pred:
            top_dominant_term = max(_constrained_top_pred.items(), key=lambda kv: kv[1])[0]
            _top_dominant_const = {
                top_dominant_term: _constrained_top_pred.get(top_dominant_term, 0)
            }
        else:
            _top_dominant_const = None
        # Correct probability to be harmonic mean of top_dominant_const and the constrained term that matched
        if _top_dominant_const is not None and constrain:
            top_dominant_const_term, top_dominant_const_score = list(_top_dominant_const.items())[0]
            matching_keys = [k for k in constrain if k in top_dominant_const_term]
            if matching_keys:
                longest_match = max(matching_keys, key=lambda x: len(x))
                longest_match_score = constrain[longest_match]
                ## If score are the same means that only text based score is given, reduce to half as this is missing the taxonomy evidence
                if longest_match_score == top_dominant_const_score:
                    constrained_score = top_dominant_const_score / 2
                else:
                    constrained_score = harmonic_mean(
                        [top_dominant_const_score, longest_match_score]
                    )
                top_dominant_const = {top_dominant_const_term: constrained_score}
                logging.debug(
                    f"Adjusted constrained score for {matching_keys}, {top_dominant_const_term},{top_dominant_const_score}: {constrained_score}"
                )
        unambiguous_constrained_predictions.append(top_dominant_const)

    ### KNN .
    # string_pattern = "Digestive system"
    # knn refinement unconstrained
    prediction_keys = [list(d.keys()) if d is not None else None for d in top_predictions]
    refined_predictions = knn_batch(
        predictions=prediction_keys,
        query_vectors=query_vector,
        params=params,
        vector_params=vector_params,
    )
    # refined_predictions = [
    #     crp if crp is None or re.search(string_pattern, list(crp.keys())[0], re.I) else None
    #     for crp in refined_predictions
    # ]

    # knn refinement constrained
    constrained_prediction_keys = [
        list(d.keys()) if d is not None else None for d in constrained_top_predictions
    ]
    constrained_refined_predictions = knn_batch(
        predictions=constrained_prediction_keys,
        query_vectors=query_vector,
        params=params,
        vector_params=vector_params,
    )
    # Set to None those refined predictions that do not match the string_pattern. predictions are List[Optional[Dict[str, float]]]
    # constrained_refined_predictions = [
    #     crp if crp is None or re.search(string_pattern, list(crp.keys())[0], re.I) else None
    #     for crp in constrained_refined_predictions
    # ]

    # load gold ontology mappings to give gold_final_prediction
    biome_herarchy_dct, biome_herarchy_dct_reversed = load_biome_herarchy_dict(
        *shared_asset_params(vector_params)
    )

    results_sequence = []
    for (
        pred,
        constr_pred,
        unambig_pred,
        unambig_constr_pred,
        refined_prediction,
        refined_constrained_prediction,
    ) in zip(
        top_predictions,
        constrained_top_predictions,
        unambiguous_predictions,
        unambiguous_constrained_predictions,
        refined_predictions,
        constrained_refined_predictions,
        strict=False,
    ):
        # define best using heristic, where the priority is on order:
        # 1. refined_constrained_prediction
        # 2. taxonomy_best_constrained_prediction
        # 3. refined_prediction
        # 4. unambig_pred
        # 5. None

        if refined_constrained_prediction is not None:
            best_heuristic = refined_constrained_prediction
        elif unambig_constr_pred is not None:
            best_heuristic = unambig_constr_pred
        elif refined_prediction is not None:
            best_heuristic = refined_prediction
        elif unambig_pred is not None:
            best_heuristic = unambig_pred
        else:
            best_heuristic = None

        def normalize_to_ontology(pred, target_ontology="GOLD"):
            if pred is None:
                return None
            if target_ontology == "GOLD":
                return {biome_herarchy_dct_reversed.get(k, k): v for k, v in pred.items()}
            else:
                return {biome_herarchy_dct.get(k, k): v for k, v in pred.items()}

        best_heuristic_gold = biome_herarchy_dct_reversed.get(
            list(best_heuristic)[0] if best_heuristic is not None else None,
            None,
        )

        if best_heuristic is not None:
            best_heuristic_gold = {best_heuristic_gold: best_heuristic[list(best_heuristic)[0]]}

        result = {
            "raw_top_predictions": normalize_to_ontology(pred, target_ontology="AMENDED"),
            "raw_unambiguous_prediction": normalize_to_ontology(
                unambig_pred, target_ontology="AMENDED"
            ),
            "constrained_top_predictions": normalize_to_ontology(
                constr_pred, target_ontology="AMENDED"
            ),
            "constrained_unambiguous_prediction": normalize_to_ontology(
                unambig_constr_pred, target_ontology="AMENDED"
            ),
            "final_selected_prediction": normalize_to_ontology(
                best_heuristic, target_ontology="AMENDED"
            ),
            "final_selected_prediction_GOLD": normalize_to_ontology(
                best_heuristic, target_ontology="GOLD"
            ),
        }
        results_sequence.append(result)

    return results_sequence


def chunked_fuzzy_prediction(
    query_vector,
    constrain,
    params: TaxonomyToBiomeParams,
    vector_params: TaxonomyToVectorParams | None = None,
):
    """Process prediction in chunks to limit memory use."""

    splits = chunked(range(query_vector.shape[0]), params.batch_size)

    results = []

    for spl in splits:
        _results = full_stack_prediction(
            query_vector[spl],
            [constrain[ix] for ix in spl],
            params=params,
            vector_params=vector_params,
        )
        results.extend(_results)
    return results


def predict_runs(
    community_vectors,
    constrain,
    params: TaxonomyToBiomeParams | None = None,
    vector_params: TaxonomyToVectorParams | None = None,
):
    """Predict lineage for samples from community vectors.

    Args:
        community_vectors: Array with shape (n_samples, dim).
        constrain: Optional per-sample prefixes from text.
        params: Model and refinement parameters.
        vector_params: Optional vectorizer parameters; when given, their
            ``local_model_dir``/repo settings are used for the shared biome
            hierarchy and tag-list assets instead of the environment.

    Returns:
        list[dict]: Prediction dicts aligned with input samples.
    """
    params = params or TaxonomyToBiomeParams()

    # Determine number of samples robustly (accept lists or numpy arrays)
    try:
        n_samples = community_vectors.shape[0]
    except Exception:
        # Fall back to len() for sequences
        try:
            n_samples = len(community_vectors)
        except Exception as exc:
            raise TypeError("Unable to determine number of samples from community_vectors") from exc

    logger.info(f"predict_runs called n_samples={n_samples}")
    # Log shape when available
    if hasattr(community_vectors, "shape"):
        logger.info(f"vectorise_run output shape={community_vectors.shape}")

    result = []
    # If community_vectors has zero feature dimension, return early
    try:
        if community_vectors.shape[1] == 0:  # no features extracted
            logger.warning(
                f"No features extracted from input; returning per-sample None placeholders n_samples={n_samples}"
            )
            # Maintain alignment with input samples: one result slot per sample
            return [None] * n_samples
    except Exception:
        # If shape not available or indexing fails, continue and let downstream code handle it
        pass

    # If no constraint array provided, create a per-sample list of None to simplify downstream indexing
    if constrain is None:
        constrain = [None] * n_samples

    result = chunked_fuzzy_prediction(
        community_vectors, constrain, params=params, vector_params=vector_params
    )
    logger.info(f"chunked_fuzzy_prediction output size={len(result)}")

    return result
