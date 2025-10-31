"""Deep learning based taxonomy prediction pipeline.

Includes model loading (lazy), vectorisation, consensus heuristics and
prediction refinement utilities.
"""
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache

import json
import re
import importlib
import logging

from typing import Any, Dict, List, Optional, Sequence, Tuple
from itertools import combinations
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from more_itertools import chunked
from tqdm import tqdm

from .taxonomy_vectorization import load_mgnify_c2v
from .config import TaxonomyToBiomeParams, TaxonomyToVectorParams
from .utils import cosine_similarity_pairwise, load_biome_herarchy_dict, get_similar_predictions,_get_hf_model_path

logger = logging.getLogger(__name__)

@lru_cache
def load_biome_tags_list():
    """Load tag list for the taxonomy classifier from HF assets.

    Returns:
        list[str]: Flat list of tag strings.
    """
    _p = TaxonomyToVectorParams()
    tags_dct_file = _get_hf_model_path(_p.hf_model, _p.model_version, "biome_tags_*.json")
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
def load_tag_biomes():
    """Map tag combinations to canonical biome lineage.

    Returns:
        tuple[dict[str, str], dict[str, tuple[set[str], int]]]:
        A mapping from tag combination to lineage, and auxiliary metadata.
    """
    biome_herarchy_dct, _ = load_biome_herarchy_dict() 
    tags_li = load_biome_tags_list()
    bioms = {x: ((set(x.split(":"))), len(x.split(":"))) for x in biome_herarchy_dct.values()}
    tag_biomes = {}
    for _prediction in tags_li:
        _n_pots = _prediction.split("|")
        for comb in generate_all_combinations(_n_pots):
            comb = set(comb)
            sels = [(k, size) for k, (se, size) in bioms.items() if len(comb) == len(comb & se)]
            if not sels:
                _comb = comb - set("Soil|Terrestrial|Non-Defined".split("|"))
                sels = [(k, size) for k, (se, size) in bioms.items() if len(_comb) == len(_comb & se)]
                if not sels:
                    _comb = _comb - {"Rhizosphere"}
                    sels = [(k, size) for k, (se, size) in bioms.items() if len(_comb) == len(_comb & se)]
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

def _get_tensorflow():
    """Lazily import TensorFlow with a clear error if unavailable."""
    try:
        return importlib.import_module("tensorflow")
    except Exception as e:
        raise RuntimeError(
            "TensorFlow is required for deep prediction but could not be imported. "
            "Install TensorFlow (for CPU-only environments: pip install tensorflow) and ensure it matches your Python version. "
            f"Original error: {e}"
        ) from e

# Load the model, including the custom loss function
@lru_cache
def load_custom_model(model_file: str | None = None):
    """Load the Keras model on demand.

    Args:
        model_file: Optional explicit file path. When None, resolve from
            Hugging Face assets for the configured model and version.

    Returns:
        Any: Compiled TensorFlow Keras model instance.

    Raises:
        RuntimeError: If loading fails or TensorFlow is unavailable.
    """
    # Resolve model path via HF hub using taxonomy_to_biome assets in the taxonomy classifier repo
    _p = TaxonomyToBiomeParams()
    model_path = _get_hf_model_path(_p.hf_model, _p.model_version, "taxonomy_to_biome_v*.model.h5")
    logger.debug(f"Loading model from file={model_path}")

    tf = _get_tensorflow()
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'focal_loss_fixed': focal_loss_fixed},
            compile=False
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='categorical_crossentropy',
            metrics=[tf.keras.metrics.AUC()]
        )
        return model
    except Exception as e:
        raise RuntimeError(
            "Failed to load the TensorFlow model. Ensure the model file is compatible with your installed TensorFlow/Keras version "
            f"and that custom objects are provided. File: {model_file}. Original error: {e}"
        ) from e


# Backwards-compatible callable used in this module
# Acts like the original model variable but resolves lazily.

def bnn_model2gg(*args, **kwargs):
    return load_custom_model()(*args, **kwargs)


def find_best_path(_prediction: str):
    _, bioms = load_tag_biomes()
    _n_pots = set(_prediction.split("|"))
    sels = [(k, size) for k, (se, size) in bioms.items() if len(_n_pots) == len(_n_pots & se)]
    sel = sorted(sels, key=lambda x: x[1])[0][0]
    return sel

def from_probs_to_pred(
    _probs,
    potential_space: List[Optional[Dict[str,float]]],
    params: TaxonomyToBiomeParams,
) -> Tuple[List[Optional[Dict[str, float]]], List[Optional[Dict[str, float]]]]:
    """Convert class probabilities into top predictions.

    Optionally constrain candidates by matching lineage prefixes from text.

    Args:
        _probs: Array of shape (n_samples, n_classes).
        potential_space: Per-sample iterable of prefix constraints or None.
        params: Thresholds and control parameters.

    Returns:
        tuple: (top_predictions, constrained_top_predictions), both lists of
        dicts mapping lineage to score, aligned to input samples.
    """
    tag_biomes,_ = load_tag_biomes()
    tags_li = load_biome_tags_list()
    
    top_predictions= []
    constrained_top_predictions= []

    
    #### WORK IN PROGRESS, HERE GET THE TOP PREDS, AND USE THAT TO REFINEMENT. USE get_similar_predictions
    for pr,_pot_space in zip(_probs,potential_space):
        if np.isnan(pr).any():
            top_p=None
            constrained_top_p=None
        else:
            # find top predictions, more than one is there is not a lot of certainty
            top_predictions_idx = get_similar_predictions(pr, params.top_prob_diff_threshold, params.top_prob_ratio_threshold)
            top_p = {tag_biomes.get(tags_li[idx],tags_li[idx]): pr[idx] for idx in top_predictions_idx}

            if _pot_space is None or len(_pot_space) == 0:
                constrained_top_p = None
            else:

                # Find matching tags in the potential (text) space
                potential_tags = {}
                for k, v in tag_biomes.items():
                    for pot,_prob in _pot_space.items():
                        if re.search("^" + re.escape(pot.strip()), v):
                            potential_tags[k] = _prob

                # If nothing matches, respect the constraint by returning it
                if len(potential_tags) == 0:
                    constrained_top_p = dict(_pot_space)
                else:
                    non_useful_tags = [ix for ix, x in enumerate(tags_li) if x not in potential_tags]
                    _pr = pr.copy()
                    _pr[non_useful_tags] = -1
                    # Use the masked probability array when computing constrained top predictions
                    constrained_top_predictions_idx = get_similar_predictions(
                        _pr,
                        diff_thresh=params.top_prob_diff_threshold,
                        ratio_thresh=params.top_prob_ratio_threshold,
                    )
                    constrained_top_p = {tag_biomes.get(tags_li[idx], tags_li[idx]): _pr[idx] for idx in constrained_top_predictions_idx}

        top_predictions.append(top_p)
        constrained_top_predictions.append(constrained_top_p)

    return top_predictions,constrained_top_predictions

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
        spl = lineage.split(':')
        for ix in range(1, len(spl) + 1):
            node = ":".join(spl[:ix])
            _node_frquencies.setdefault(node, []).append(count)
            
    node_frequencies = {k: sum(v)  for k, v in _node_frquencies.items()}
    _filtered = [
        (k, v)
        for k, v in node_frequencies.items()
        if v > dominance_threshold and k != '' and len(k.split(":")) >= best_lineage_min_depth # Only consider nodes with sufficient depth
    ]
    sorted_passed = sorted(_filtered, key=lambda x: len(x[0].split(":")), reverse=True)
    top_dominant = sorted_passed[0] if sorted_passed else [co.index[0],co.iloc[0]]
    
    # Claculate score based on mean of score for each lineage containing the top dominant node
    top_dominant_score = None
    if top_dominant is not None:
        top_dominant_score = np.sum([v for k, v in co.items() if top_dominant[0] in k ])
        top_dominant = {top_dominant[0]: top_dominant_score}

    return node_frequencies, sorted_passed, top_dominant

def refine_predictions_knn_batch(
    predictions: List[Optional[List[Any]]],
    query_vectors: np.ndarray,
    params: TaxonomyToBiomeParams,
) -> List[Optional[Dict[str, float]]]:
    """DEPRECATED"""
    
    """Refine deep model predictions using KNN in the vector space (batch version).

    This version processes multiple predictions in parallel by grouping queries
    by their unique predicted prefix. Each unique prefix is refined once, and
    results are mapped back to their original order.

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
    List[Optional[Dict[str, float]]]
        A list of refined predictions aligned with the input order.
    """

    logger.debug("DEPRECATED: Starting batch KNN refinement of predictions")
    
    # Prepare result list in input order
    results: List[Optional[Dict[str, float]]] = [None] * len(predictions)
    if params.k_knn <= 0:
        logger.debug("k_knn <= 0; skipping KNN refinement")
        return results

    # Load MGnify sample vectors and metadata
    mgnify_sample_vectors, mgnify_meta = load_mgnify_c2v(
        model_name=params.hf_model,
        model_version=params.model_version,
    )

    max_per_study = max(1, params.k_knn // 3)

    # Normalize predictions: None -> empty string

    # Group query indices by unique prediction key
    groups: Dict[str, List[int]] = defaultdict(list)
    for ix, pred in enumerate(predictions):
        key = "|".join(sorted(pred)) if pred else None
        if key:
            groups[key].append(ix)


    # Process each group once
    for pred_key, indices in groups.items():
        prediction = pred_key if pred_key else ''

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
        argsort_sims = np.argsort(sims)

        # Process each query in this group
        for local_ix, ass in enumerate(argsort_sims):
            global_ix = indices[local_ix]

            # _co = _subject_df.iloc[ass[-params.k_knn:]]["BIOME_AMEND"].value_counts()

            # Take similarity scores for this query
            sim_scores = sims[local_ix]

            # Sort indices by descending similarity
            sorted_ix = np.argsort(sim_scores)[::-1]

            # Subset dataframe with similarity order
            top_df = _subject_df.iloc[sorted_ix].copy()
            top_df["SIM"] = sim_scores[sorted_ix]
            
            top_df_limited = top_df.groupby("STUDY_ID", group_keys=False).head(max_per_study)

            # Now take the top k_knn overall, after applying the per-study cap
            selected_df = top_df_limited.head(params.k_knn)

            # Compute frequencies for the selected subset
            _co = selected_df["BIOME_AMEND"].value_counts() / selected_df.shape[0]
            _, _, top_dominant = get_unanbigious_prediction(_co, dominance_threshold=params.dominance_threshold)

            if not top_dominant:
                refined_prediction = None
            else:
                refined_prediction = top_dominant

            results[global_ix] = refined_prediction

    return results



np.seterr(
    divide="ignore", invalid="ignore"
)  # handle bad files == divition by zero error


def full_stack_prediction(query_vector, constrains, params:TaxonomyToBiomeParams) -> List[Dict[str, Any]]:
    """Predict biome using deep model and KNN refinement.

    Applies optional constraints from text and returns per-sample dicts with
    raw, constrained, refined, and final selections.
    """
    # prediction baded on deep learning model
    logger.info("Starting full stack prediction")
    deep_l_probs = bnn_model2gg(query_vector).numpy()

    top_predictions,constrained_top_predictions = from_probs_to_pred(deep_l_probs, potential_space=constrains,params=params)

    # Get unambiguous predictions
    unambiguous_predictions = []
    unambiguous_constrained_predictions = []
    for _top_pred,_constrained_top_pred in zip(top_predictions,constrained_top_predictions):
        _, _, top_dominant = get_unanbigious_prediction(pd.Series(_top_pred), dominance_threshold=params.dominance_threshold)
        unambiguous_predictions.append(top_dominant)

        if _constrained_top_pred is None:
            unambiguous_constrained_predictions.append(None)
            continue
        
        _, _, top_dominant_const = get_unanbigious_prediction(pd.Series(_constrained_top_pred), dominance_threshold=params.dominance_threshold)
        unambiguous_constrained_predictions.append(top_dominant_const)
    
    ### KNN refinement. 
    # string_pattern = "Digestive system"
    # knn refinement unconstrained
    prediction_keys = [list(d.keys()) if d is not None else None for d in top_predictions]
    refined_predictions = refine_predictions_knn_batch( predictions=prediction_keys, query_vectors=query_vector, params=params)
    # refined_predictions = [
    #     crp if crp is None or re.search(string_pattern, list(crp.keys())[0], re.I) else None
    #     for crp in refined_predictions
    # ]

    # knn refinement constrained
    constrained_prediction_keys = [list(d.keys()) if d is not None else None for d in constrained_top_predictions]
    constrained_refined_predictions = refine_predictions_knn_batch( predictions=constrained_prediction_keys, query_vectors=query_vector, params=params)
    # Set to None those refined predictions that do not match the string_pattern. predictions are List[Optional[Dict[str, float]]]
    # constrained_refined_predictions = [
    #     crp if crp is None or re.search(string_pattern, list(crp.keys())[0], re.I) else None
    #     for crp in constrained_refined_predictions
    # ]

    # load gold ontology mappings to give gold_final_prediction
    biome_herarchy_dct, biome_herarchy_dct_reversed = load_biome_herarchy_dict() 

    results_sequence = []
    for pred, constr_pred, unambig_pred, unambig_constr_pred, refined_prediction, refined_constrained_prediction in zip(
        top_predictions,
        constrained_top_predictions,
        unambiguous_predictions,
        unambiguous_constrained_predictions,
        refined_predictions,
        constrained_refined_predictions,
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

        def normalize_to_amended_ontology(pred,target_ontology="GOLD"):
            if pred is None:
                return None
            if target_ontology=="GOLD":
                return {biome_herarchy_dct_reversed.get(k, k):v for k, v in pred.items()}
            else:
                return {biome_herarchy_dct.get(k, k):v for k, v in pred.items()}



        best_heuristic_gold = biome_herarchy_dct_reversed.get(
            list(best_heuristic)[0] if best_heuristic is not None else None,
            None,
        )
        
        if best_heuristic is not None:
            best_heuristic_gold = {best_heuristic_gold: best_heuristic[list(best_heuristic)[0]]}
        
        result = {
            "raw_top_predictions": normalize_to_amended_ontology(pred, target_ontology="AMENDED"),
            "raw_unambiguous_prediction": normalize_to_amended_ontology(unambig_pred, target_ontology="AMENDED"),
            "raw_refined_prediction": normalize_to_amended_ontology(refined_prediction, target_ontology="AMENDED"),
            "constrained_top_predictions": normalize_to_amended_ontology(constr_pred, target_ontology="AMENDED"),
            "constrained_unambiguous_prediction": normalize_to_amended_ontology(unambig_constr_pred, target_ontology="AMENDED"),
            "constrained_refined_prediction": normalize_to_amended_ontology(refined_constrained_prediction, target_ontology="AMENDED"),
            "final_selected_prediction": normalize_to_amended_ontology(best_heuristic, target_ontology="AMENDED"),
        }
        results_sequence.append(result)

    return results_sequence

def chunked_fuzzy_prediction(query_vector, constrain, params:TaxonomyToBiomeParams):
    """Process prediction in chunks to limit memory use."""

    splits = chunked(range(query_vector.shape[0]), params.batch_size)

    results = []

    for spl in splits:
        _results = full_stack_prediction(
            query_vector[spl], [constrain[ix] for ix in spl], params=params
        )
        results.extend(_results)
    return results

def predict_runs(
    community_vectors,
    constrain,
    params: TaxonomyToBiomeParams = TaxonomyToBiomeParams(),
):
    """Predict lineage for samples from community vectors.

    Args:
        community_vectors: Array with shape (n_samples, dim).
        constrain: Optional per-sample prefixes from text.
        params: Model and refinement parameters.

    Returns:
        list[dict]: Prediction dicts aligned with input samples.
    """
    # Determine number of samples robustly (accept lists or numpy arrays)
    try:
        n_samples = community_vectors.shape[0]
    except Exception:
        # Fall back to len() for sequences
        try:
            n_samples = len(community_vectors)
        except Exception:
            raise TypeError("Unable to determine number of samples from community_vectors")

    logger.info(f"predict_runs called n_samples={n_samples}")
    # Log shape when available
    if hasattr(community_vectors, "shape"):
        logger.info(f"vectorise_run output shape={community_vectors.shape}")

    result = []
    # If community_vectors has zero feature dimension, return early
    try:
        if community_vectors.shape[1] == 0:  # no features extracted
            logger.warning(f"No features extracted from input; returning empty DataFrame n_samples={n_samples}")
            return result
    except Exception:
        # If shape not available or indexing fails, continue and let downstream code handle it
        pass

    # If no constraint array provided, create a per-sample list of None to simplify downstream indexing
    if constrain is None:
        constrain = [None] * n_samples

    result = chunked_fuzzy_prediction(community_vectors, constrain, params=params)
    logger.info(f"chunked_fuzzy_prediction output size={len(result)}")

    return result
