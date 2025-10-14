"""Deep learning based taxonomy prediction pipeline.

Includes model loading (lazy), vectorisation, consensus heuristics and
prediction refinement utilities.
"""
from __future__ import annotations
from functools import lru_cache

from .api import Community2vec



import os
import json
import re
import importlib
import logging
from typing import List, Sequence

import numpy as np
import pandas as pd
import sys
import json
import numpy as np
import pandas as pd
from .utils import cosine_similarity_pairwise, get_path,tax_annotations_from_file, load_biome_herarchy_dict
from .taxonomy_vectorization import genus_from_edges_subgraph, genre_to_taxonomy_vectorization
import networkx as nx
from more_itertools import chunked

logger = logging.getLogger(__name__)


from tqdm import tqdm

from . import config


from .taxonomy_vectorization import load_mgnify_c2v
from . import model_registry

biome_herarchy_dct = load_biome_herarchy_dict()

TAG = 'taxonomy_prediction'

DATA_DIR = f"{config.datadir}/{TAG}"
TMP_DIR = f"{DATA_DIR}/temp"
os.makedirs(TMP_DIR,exist_ok=True)

from .utils import cosine_similarity_pairwise


from .taxonomy_vectorization import load_mgnify_c2v



@lru_cache
def load_biome_tags_list():
    tags_dct_file = get_path("resources/taxonomy/biome_tags_dct_file.json")
    if not os.path.exists(tags_dct_file):
        raise FileNotFoundError(
            f"Biome tags dictionary file not found: {tags_dct_file}\n"
            "Download using trapiche-download-models"
            )
    logger.debug(f"Loading biome tags dictionary from file={tags_dct_file}")
    with open(tags_dct_file) as h:
        tags_dct = json.load(h)
    tags_li = list(tags_dct)
    return tags_li

from itertools import combinations

def generate_all_combinations(s: Sequence[str]):
    """Generate all combinations (powerset) of an iterable sequence."""
    result = []
    for r in range(len(s) + 1):
        result.extend(combinations(s, r))
    return result

@lru_cache
def load_tag_biomes():
    biome_herarchy_dct = load_biome_herarchy_dict() 
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
    # Your implementation of the focal loss
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
    """Load and compile the Keras model on demand with robust error handling.
    model_file: optional explicit path; if None, resolve via registry/legacy path.
    """
    model_path = get_path("models/taxonomy/taxonomy_to_biome/1.0/taxonomy_to_biome_v1.0.model.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}\n"
                                "Download using trapiche-download-models")
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


def pc_deviation_consensus(pr, three=1.0):
    tag_biomes,_ = load_tag_biomes()
    tags_li = load_biome_tags_list()
    asp = np.argsort(pr)
    topp = pr[asp[-1]]
    _dct = {tags_li[ix]: pr[ix] for ix in asp if pr[ix] / topp >= three}
    _dct.update({tags_li[asp[-1]]:pr[asp[-1]]})
    lins = list(_dct)
    consensus = set(lins[0].split("|"))
    for lin in lins:
        consensus &= set(lin.split("|"))

    _c = "|".join(sorted(consensus))
    res = tag_biomes[_c]
    return res, sum(_dct.values())



def find_best_path(_prediction: str):
    _, bioms = load_tag_biomes()
    _n_pots = set(_prediction.split("|"))
    sels = [(k, size) for k, (se, size) in bioms.items() if len(_n_pots) == len(_n_pots & se)]
    sel = sorted(sels, key=lambda x: x[1])[0][0]
    return sel

def from_probs_to_pred(_probs, potential_space=None):
    """Predict biome, optionally constrained to a candidate potential space (mixed prediction)."""
    tag_biomes,_ = load_tag_biomes()
    tags_li = load_biome_tags_list()
    result = []
    if potential_space is None:
        potential_space = [[] for _ in _probs]
    for pr,_pot_space in zip(_probs,potential_space):
        if np.isnan(pr).any():
            du = (None, 0.0)
        else:
            if len(_pot_space) == 0:
                du = pc_deviation_consensus(pr)
            else:
                potential_tags = {k for k, v in tag_biomes.items() if re.search("|".join(_pot_space), v)}
                if len(potential_tags) == 0:
                    du = ("|".join(_pot_space), 0.0)
                else:
                    non_useful_tags = [ix for ix, x in enumerate(tags_li) if x not in potential_tags]
                    _pr = pr.copy()
                    _pr[non_useful_tags] = 0
                    du = pc_deviation_consensus(_pr)
        result.append(du)

    return result

def get_lineage_frquencies(co):  # spelling retained for backwards compatibility
    """ Calculate fuzzy array for each sample
    The idea is that each node in the BIOME_AMEND space is a fuzzy category that can be calculated via the frequency of the node in the lineage of the KNN samples:
    """
    """ Function to calculate the requency of each node in the lineages of the knn samples
    """
    total_samples = co.sum()
    _node_frquencies = {}
    for lineage, count in co.items():
        spl = lineage.split(':')
        for ix in range(1, len(spl) + 1):
            node = ":".join(spl[:ix])
            _node_frquencies.setdefault(node, []).append(count)
    node_frequencies = {k: sum(v) / total_samples for k, v in _node_frquencies.items()}
    return node_frequencies

def refine_predictions_knn(prediction, query_vector, full_subject_df, tru_column='BIOME_AMEND', k_knn=10, dominance_threshold=0.5, vector_space='g'):
    """ Function that given a previous prediction from the deepL model, finds close relatives
    """
    mgnify_sample_vectors, _ = load_mgnify_c2v()
    if prediction is None:
        prediction = ''
    _subject_df = full_subject_df[full_subject_df[tru_column].str.contains(prediction)]
    if vector_space == 'g':
        subject_vector = mgnify_sample_vectors.loc[_subject_df.index]
    else:
        raise ValueError("Only vector_space='g' is supported in refine_predictions_knn")
    sims = cosine_similarity_pairwise(query_vector,subject_vector)
    sims[np.isnan(sims)] = 0
    argsort_sims = np.argsort(sims)
    
    result =  []
    
    for ix, ass in enumerate(argsort_sims):
        _co = _subject_df.iloc[ass[-k_knn:]][tru_column].value_counts()
        _co.index = [x.replace(prediction, '') for x in _co.index]
        _node_freqs = get_lineage_frquencies(_co)
        _filtered = [(k, len(k.split(":"))) for k, v in _node_freqs.items() if v > dominance_threshold and k != '']
        _so = sorted(_filtered, key=lambda x: x[1], reverse=True)
        result.append(f"{prediction}{'' if len(_so)==0 else _so[0][0]}")
    return result



np.seterr(
    divide="ignore", invalid="ignore"
)  # handle bad files == divition by zero error


def full_stack_prediction(query_vector, constrain,k_knn=10, dominance_threshold=0.5, vector_space="g"):
    """Function for prediction of biome based on taxonomic compositon"""
    # prediction baded on deep learning model
    # deep_l_probs = tflite_prediction(taxo_qunat_interpreter, query_vector)
    _, mgnify_sample_vectors_metadata = load_mgnify_c2v()
    tag_biomes,_ = load_tag_biomes()
    tags_li = load_biome_tags_list()

    mgnify_sample_vectors_metadata['BIOME_AMEND'] = mgnify_sample_vectors_metadata.LINEAGE.map(lambda x: biome_herarchy_dct.get(x, x))
    
    deep_l_probs = bnn_model2gg(query_vector).numpy()

    predicted_lineages = from_probs_to_pred(deep_l_probs, potential_space=constrain)

    pred_df = pd.DataFrame(
        predicted_lineages, columns=["lineage_prediction", "lineage_prediction_probability"]
    )

    # refinement phase
    refined = [None] * pred_df.shape[0]

    for pred, gr in pred_df.groupby("lineage_prediction"):
        query_array = query_vector[gr.index]
        refs = refine_predictions_knn(
            pred, query_array, mgnify_sample_vectors_metadata, k_knn=k_knn, dominance_threshold=dominance_threshold, vector_space=vector_space
        )
        for ix, pp in zip(gr.index, refs):
            refined[ix] = pp
    pred_df["refined_prediction"] = refined
    pred_df["prediction_non_constrained"] = [tag_biomes.get(tags_li[idx], None) for idx in np.argsort(deep_l_probs)[:, -1]]

    for ix, t in enumerate(tags_li):
        pred_df[t] = deep_l_probs[:, ix]

    return pred_df

def chunked_fuzzy_prediction(query_vector, constrain,k_knn=10, dominance_threshold=0.5, batch_size=200, vector_space="g"):
    """Process prediction in chunks to limit memory usage."""

    splits = chunked(range(query_vector.shape[0]), batch_size)

    results = []

    for spl in tqdm(splits, desc=TAG):
        _results = full_stack_prediction(
            query_vector[spl], [constrain[ix] for ix in spl],k_knn=k_knn,dominance_threshold=dominance_threshold, vector_space=vector_space
        )
        results.append(_results)
    return pd.concat(results).reset_index()

def predict_runs(
    community_vectors,
    return_full_preds=False,
    constrain=None,
    batch_size=200,
    k_knn=10,
    dominance_threshold=0.5,
):
    """Predict lineage of runs based on multiple taxonomy files (diamond/SSU/LSU mix)."""
    logger.info(f"predict_runs called n_samples={len(community_vectors)} return_full_preds={return_full_preds}")

    logger.info(f"vectorise_run output shape={community_vectors.shape}")

    if community_vectors.shape[1] == 0:  # no features extracted
        logger.warning(f"No features extracted from input; returning empty DataFrame n_samples={len(community_vectors)}")
        empty_df = pd.DataFrame({
            "refined_prediction": [None]*len(community_vectors),
            "lineage_prediction_probability": [np.nan]*len(community_vectors),
            "lineage_prediction": [None]*len(community_vectors),
            "prediction_non_constrained": [None]*len(community_vectors),
        })
        return (empty_df if return_full_preds else empty_df[["refined_prediction", "lineage_prediction_probability"]]), community_vectors

    if constrain is None:
        constrain = [[] for _ in community_vectors]

    result = chunked_fuzzy_prediction(community_vectors, constrain, k_knn=k_knn, dominance_threshold=dominance_threshold, batch_size=batch_size)
    logger.info(f"chunked_fuzzy_prediction output shape={result.shape if hasattr(result, 'shape') else None}")
    logger.debug(f"result columns={getattr(result, 'columns', None)} head={result.head().to_dict() if hasattr(result, 'head') else None}")

    if not return_full_preds:
        result = result[["refined_prediction", "lineage_prediction_probability"]]
    logger.info(f"predict_runs returning result shape={result.shape if hasattr(result, 'shape') else None}")
    return result, community_vectors
