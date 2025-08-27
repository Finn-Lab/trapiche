"""Deep learning based taxonomy prediction pipeline.

Includes model loading (lazy), vectorisation, consensus heuristics and
prediction refinement utilities.
"""
from __future__ import annotations

# %% auto 0
__all__ = ['TAG', 'DATA_DIR', 'TMP_DIR', 'comm2vecs', 'comm2vecs_metadata', 'k', 'min_projects_vote', 'dominance_thres', 'k2',
           'dominance_thres2', 'tags_dct_file', 'tags_li', 'bioms', 'tag_biomes', 'tnp_core_df_file', 'slim_core_df',
           'best_params', 'final_model_file', 'bnn_model2gg', 'get_bnn_model', 'tflite_model_quant_file', 'generate_all_combinations',
           'focal_loss_fixed', 'load_custom_model', 'pc_deviation_consensus', 'find_best_path', 'from_probs_to_pred',
           'get_lineage_frquencies', 'refine_predictions_knn', 'full_stack_prediction', 'chunked_fuzzy_prediction',
           'vectorise_run', 'predict_runs']

# %% ../nbs/01.00.04_deep_pred.ipynb 5
import os
import json
import re
import importlib
from typing import List, Sequence

import numpy as np
import pandas as pd
try:  # prefer external helper if available
    from more_itertools import chunked  # type: ignore
except Exception:  # simple local fallback
    def chunked(iterable, size):
        buf = []
        for x in iterable:
            buf.append(x)
            if len(buf) == size:
                yield buf
                buf = []
        if buf:
            yield buf
from tqdm import tqdm

from . import config

# %% ../nbs/01.00.04_deep_pred.ipynb 6
from .goldOntologyAmendments import gold_categories,biome_graph,biome_original_graph

# %% ../nbs/01.00.04_deep_pred.ipynb 7
from .biome2vec import load_mgnify_c2v
from . import model_registry
from .goldOntologyAmendments import biome_herarchy_dct

# %% ../nbs/01.00.04_deep_pred.ipynb 8
TAG = 'deep_pred'

# %% ../nbs/01.00.04_deep_pred.ipynb 9
DATA_DIR = f"{config.datadir}/{TAG}"
TMP_DIR = f"{DATA_DIR}/temp"
os.makedirs(TMP_DIR,exist_ok=True)

# %% ../nbs/01.00.04_deep_pred.ipynb 10
from . import taxonomyTree
from .baseData import analysis_df
# from trapiche.baseData import load_taxonomy_sets
# from trapiche.baseData import RESULTS_BASE_DIR

# %% ../nbs/01.00.04_deep_pred.ipynb 11
from .utils import cosine_similarity,jaccard_similarity,cosine_similarity_pairwise

# %% ../nbs/01.00.04_deep_pred.ipynb 14
from .biome2vec import taxo_ids
from .taxonomyTree import taxonomy_graph
from . import baseData

# %% ../nbs/01.00.04_deep_pred.ipynb 15
comm2vecs, comm2vecs_metadata = load_mgnify_c2v()

# %% ../nbs/01.00.04_deep_pred.ipynb 21
comm2vecs_metadata['BIOME_AMEND'] = comm2vecs_metadata.LINEAGE.map(lambda x: biome_herarchy_dct.get(x, x))

# %% ../nbs/01.00.04_deep_pred.ipynb 24
""" PARAMS
"""
k = 100
min_projects_vote = 3  # minimum number of project votes for consideration
dominance_thres = 0.5  # fraction threshold in top k for consideration
k2 = 33
dominance_thres2 = 0.5

# %% ../nbs/01.00.04_deep_pred.ipynb 26
from .biome2vec import comm2vecs_file,load_mgnify_c2v

# %% ../nbs/01.00.04_deep_pred.ipynb 40
from .goldOntologyAmendments import biome_herarchy_dct

# %% ../nbs/01.00.04_deep_pred.ipynb 62
# set_piority_terms = ({xx for x in core_df[~core_df.min_annots_amended.isna()].min_annots_amended.unique() for xx in x.split(":")[-2:]}-{'Environmental','Host-associated','Gastrointestinal tract'})|{'Agricultural', 'Agricultural field', 'Amphibia', 'Asphalt lakes', 'Boreal forest', 'Bryozoans', 'Clay', 'Contaminated', 'Crop', 'Desert', 'Forest soil', 'Fossil', 'Grasslands', 'Loam', 'Lymph nodes', 'Milk', 'Mine', 'Mine drainage', 'Nasopharyngeal', 'Nervous system', 'Oil-contaminated', 'Permafrost', 'Pulmonary system', 'Rhizome', 'Rock-dwelling', 'Sand', 'Shrubland', 'Silt', 'Tar', 'Tropical rainforest', 'Tunicates', 'Uranium contaminated', 'Urethra', 'Vagina', 'Wetlands'}

# %% ../nbs/01.00.04_deep_pred.ipynb 64
tags_dct_file =f"{DATA_DIR}/tags_dct_file.json"

# %% ../nbs/01.00.04_deep_pred.ipynb 66
with open(tags_dct_file) as h:
    tags_dct = json.load(h)

# %% ../nbs/01.00.04_deep_pred.ipynb 67
tags_li = list(tags_dct)

# %% ../nbs/01.00.04_deep_pred.ipynb 68
from itertools import combinations

# %% ../nbs/01.00.04_deep_pred.ipynb 69
def generate_all_combinations(s: Sequence[str]):
    """Generate all combinations (powerset) of an iterable sequence."""
    result = []
    for r in range(len(s) + 1):
        result.extend(combinations(s, r))
    return result

# %% ../nbs/01.00.04_deep_pred.ipynb 70
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

# %% ../nbs/01.00.04_deep_pred.ipynb 85
tnp_core_df_file = f"{DATA_DIR}/core_df_file_2.tsv"

# %% ../nbs/01.00.04_deep_pred.ipynb 87
slim_core_df = pd.read_csv(tnp_core_df_file, sep='\t', index_col='SAMPLE_ID')

# %% ../nbs/01.00.04_deep_pred.ipynb 120
# best_params['epochs'] = hist.val_aucpr.argmax()
best_params = {
    'complex': 450.0,
    'gamma': 0.5,
    'for_validation': 'NoAug.histo.json',
    'random_state': 1.5,
    'val_aucpr': 0.546550914645195,
    'test_aucpr': 0.5262038260698318,
    'epochs': 87,
}

# %% ../nbs/01.00.04_deep_pred.ipynb 124
final_model_file = f"{DATA_DIR}/full_final_taxonomy.model.keras"  # legacy path if packaged

# %% ../nbs/01.00.04_deep_pred.ipynb 132
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

def _resolve_final_model_file():
    if os.path.exists(final_model_file):
        return final_model_file
    # Use registry (auto download)
    p = model_registry.get_model_path("full_final_taxonomy.model.keras", auto_download=True)
    return str(p)


def load_custom_model(model_file: str | None = None):
    """Load and compile the Keras model on demand with robust error handling.
    model_file: optional explicit path; if None, resolve via registry/legacy path.
    """
    if model_file is None:
        model_file = _resolve_final_model_file()
    if not os.path.exists(model_file):  # defensive
        raise FileNotFoundError(f"Model file not found after resolution: {model_file}")
    tf = _get_tensorflow()
    try:
        model = tf.keras.models.load_model(
            model_file,
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

# Cached accessor for the model
_bnn_model_cache = None

def get_bnn_model():
    """Get a cached instance of the model, loading it on first use."""
    global _bnn_model_cache
    if _bnn_model_cache is None:
        _bnn_model_cache = load_custom_model()
    return _bnn_model_cache

# Backwards-compatible callable used in this module
# Acts like the original model variable but resolves lazily.

def bnn_model2gg(*args, **kwargs):
    return get_bnn_model()(*args, **kwargs)

# %% ../nbs/01.00.04_deep_pred.ipynb 135
tflite_model_quant_file = f"{DATA_DIR}/taxonomy_quant_model.tflite"

# %% ../nbs/01.00.04_deep_pred.ipynb 154
# def pc_deviation_consensus(pr,three = 0.66):
def pc_deviation_consensus(pr, three=1.0):
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

# %% ../nbs/01.00.04_deep_pred.ipynb 159
bioms = {x: ((set(x.split(":"))), len(x.split(":"))) for x in biome_herarchy_dct.values()}

def find_best_path(_prediction: str):
    _n_pots = set(_prediction.split("|"))
    sels = [(k, size) for k, (se, size) in bioms.items() if len(_n_pots) == len(_n_pots & se)]
    sel = sorted(sels, key=lambda x: x[1])[0][0]
    return sel

# %% ../nbs/01.00.04_deep_pred.ipynb 160
def from_probs_to_pred(_probs, potential_space=None):
    """Predict biome, optionally constrained to a candidate potential space (mixed prediction)."""
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

# %% ../nbs/01.00.04_deep_pred.ipynb 163
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

# %% ../nbs/01.00.04_deep_pred.ipynb 164
# def refine_predictions_knn(prediction,query_vector,full_subject_df,tru_column = 'BIOME_AMEND',k=10,vector_space='g'):
def refine_predictions_knn(prediction, query_vector, full_subject_df, tru_column='BIOME_AMEND', k=10, vector_space='g'):
    """ Function that given a previous prediction from the deepL model, finds close relatives
    """
    if prediction is None:
        prediction = ''
    _subject_df = full_subject_df[full_subject_df[tru_column].str.contains(prediction)]
    if vector_space == 'g':
        subject_vector = comm2vecs.loc[_subject_df.index]
    else:
        raise ValueError("Only vector_space='g' is supported in refine_predictions_knn")
    sims = cosine_similarity_pairwise(query_vector,subject_vector)
    sims[np.isnan(sims)] = 0
    argsort_sims = np.argsort(sims)
    
    result =  []
    
    for ix, ass in enumerate(argsort_sims):
        _co = _subject_df.iloc[ass[-k:]][tru_column].value_counts()
        _co.index = [x.replace(prediction, '') for x in _co.index]
        _node_freqs = get_lineage_frquencies(_co)
        _filtered = [(k, len(k.split(":"))) for k, v in _node_freqs.items() if v > 0.5 and k != '']
        _so = sorted(_filtered, key=lambda x: x[1], reverse=True)
        result.append(f"{prediction}{'' if len(_so)==0 else _so[0][0]}")
    return result

# %% ../nbs/01.00.04_deep_pred.ipynb 189
import sys
import json
import numpy as np
import pandas as pd
from .utils import cosine_similarity_pairwise
from .baseData import tax_annotations_from_file
from .biome2vec import genus_from_edges_subgraph, genre_to_comm2vec
import networkx as nx
from .goldOntologyAmendments import biome_graph


# %% ../nbs/01.00.04_deep_pred.ipynb 190
np.seterr(
    divide="ignore", invalid="ignore"
)  # handle bad files == divition by zero error


# %% ../nbs/01.00.04_deep_pred.ipynb 191
def full_stack_prediction(query_vector, constrain, vector_space="g"):
    """Function for prediction of biome based on taxonomic compositon"""
    # prediction baded on deep learning model
    # deep_l_probs = tflite_prediction(taxo_qunat_interpreter, query_vector)
    deep_l_probs = bnn_model2gg(query_vector).numpy()

    predicted_lineages = from_probs_to_pred(deep_l_probs, potential_space=constrain)

    pred_df = pd.DataFrame(
        predicted_lineages, columns=["lineage_pred", "lineage_pred_prob"]
    )

    # refinement phase
    refined = [None] * pred_df.shape[0]

    for pred, gr in pred_df.groupby("lineage_pred"):
        query_array = query_vector[gr.index]
        # refs = refine_predictions_knn(pred,query_array,slim_core_df,vector_space=vector_space)
        refs = refine_predictions_knn(
            pred, query_array, comm2vecs_metadata, vector_space=vector_space
        )
        for ix, pp in zip(gr.index, refs):
            refined[ix] = pp
    pred_df["refined_prediction"] = refined
    pred_df["unbiased_taxo_prediction"] = [tag_biomes.get(tags_li[idx], None) for idx in np.argsort(deep_l_probs)[:, -1]]
    

    for ix, t in enumerate(tags_li):
        pred_df[t] = deep_l_probs[:, ix]

    return pred_df

# %% ../nbs/01.00.04_deep_pred.ipynb 192
def chunked_fuzzy_prediction(query_vector, constrain, chunk_size=200, vector_space="g"):
    """Process prediction in chunks to limit memory usage."""

    splits = chunked(range(query_vector.shape[0]), chunk_size)

    results = []

    for spl in tqdm(splits, desc=TAG):
        _results = full_stack_prediction(
            query_vector[spl], [constrain[ix] for ix in spl], vector_space=vector_space
        )
        results.append(_results)
    return pd.concat(results).reset_index()

# %% ../nbs/01.00.04_deep_pred.ipynb 193
def vectorise_run(list_of_tax_files, vector_space="g"):
    """Vectorise a run based on multiple taxonomy files (e.g. diamond + LSU + SSU)."""
    samples_annots = {}
    for ix, samp_files in enumerate(list_of_tax_files):
        if not samp_files:  # skip empty lists
            continue
        for f in samp_files:
            try:
                d = tax_annotations_from_file(f)
            except Exception as e:  # defensive
                print(f"Failed to parse taxonomy file {f}: {e}")
                d = None
            if d:
                samples_annots.setdefault(ix, []).extend(d)
    samples_genus = {k: genus_from_edges_subgraph(e) for k, e in samples_annots.items()}
    samples_vecs = {k: genre_to_comm2vec(gs) for k, gs in samples_genus.items()}
    if not samples_vecs:  # no vectors parsed
        # Return an empty array with zero rows; caller should handle
        return np.zeros((len(list_of_tax_files), 0))
    # Ensure every sample index has a vector (fill missing with zeros of correct length)
    first_vec = next(iter(samples_vecs.values()))
    vec_len = len(first_vec)
    ordered = []
    for i in range(len(list_of_tax_files)):
        if i in samples_vecs:
            ordered.append(samples_vecs[i])
        else:
            ordered.append(np.zeros(vec_len))
    vec = np.stack(ordered, axis=0)
    return vec

# %% ../nbs/01.00.04_deep_pred.ipynb 194
def predict_runs(
    list_of_list,
    vector_space="g",
    return_full_preds=False,
    constrain=None,
):
    """Predict lineage of runs based on multiple taxonomy files (diamond/SSU/LSU mix)."""
    # _vec_size = vec_size*(1 if vector_space=='g' else 2)

    # full_vec= np.zeros((len(list_of_list),_vec_size))
    full_vec = vectorise_run(list_of_list)

    if full_vec.shape[1] == 0:  # no features extracted
        empty_df = pd.DataFrame({
            "refined_prediction": [None]*len(list_of_list),
            "lineage_pred_prob": [np.nan]*len(list_of_list),
            "lineage_pred": [None]*len(list_of_list),
            "unbiased_taxo_prediction": [None]*len(list_of_list),
        })
        return (empty_df if return_full_preds else empty_df[["refined_prediction", "lineage_pred_prob"]]), full_vec

    if constrain is None:
        constrain = [[] for _ in list_of_list]

    result = chunked_fuzzy_prediction(full_vec, constrain, vector_space=vector_space)

    if not return_full_preds:
        result = result[["refined_prediction", "lineage_pred_prob"]]
    return result, full_vec
