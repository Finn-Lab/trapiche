"""Biome2Vec embedding utilities.

Load pre-trained word2vec style embeddings and provide helpers to aggregate
taxonomy subgraphs into community vectors.
"""
from __future__ import annotations

__all__ = ['TAG', 'DATA_DIR', 'TMP_DIR', 'SG', 'biome2vec_file', 'model_vocab_file', 'vec_file', 'taxo_ids', 'vec',
           'comm2vecs_file', 'comm2vecs_metadata_file', 'load_biome2vec', 'sentence_vectorization',
           'genus_from_edges_subgraph', 'get_terminals', 'get_mean', 'genre_to_comm2vec', 'load_mgnify_c2v']

import json
import os
import numpy as np
import pandas as pd

from . import config, taxonomyTree, model_registry

import networkx as nx
from gensim.models import Word2Vec
from .utils import cosine_similarity


TAG = "biome2vec"


DATA_DIR = f"{config.datadir}/{TAG}"
TMP_DIR = f"{DATA_DIR}/temp"
os.makedirs(TMP_DIR, exist_ok=True)




SG = 1  # 1 for skip gram


biome2vec_file = f"{DATA_DIR}/word2vec.sg_{SG}_full.model"


model_vocab_file = f"{DATA_DIR}/model_vocab.json"


with open(model_vocab_file) as h:
    model_vocab = json.load(h)


def load_biome2vec():
    """Load the word2vec model for biome embeddings."""
    return Word2Vec.load(biome2vec_file)


vec_file = f"{biome2vec_file}.wv.vectors.npy"


taxo_ids = {x: ix for ix, x in enumerate(taxonomyTree.taxonomy_graph.nodes)}


model_vocab_file = f"{DATA_DIR}/model_vocab.json"


with open(model_vocab_file) as h:
    model_vocab = json.load(h)


vec_file = f"{biome2vec_file}.wv.vectors.npy"


vec = np.load(vec_file)




def sentence_vectorization(terminals):
    """Compute mean vector from terminal nodes of a subgraph (community)."""
    # tax ='Laterosporus'
    tix_ = [taxo_ids.get(tax) for tax in terminals]
    tixs = [x for x in tix_ if x != None]
    v_ixs_ = [model_vocab.get(str(tix)) for tix in tixs]
    v_ixs = [x for x in v_ixs_ if x != None]
    _t = [vec[v_ix] for v_ix in v_ixs]

    # _t = [x for x in terminals if x in embs]
    mean = np.mean(_t, axis=0)
    return mean


def genus_from_edges_subgraph(edges: list):
    """Extract genera from taxonomy subgraph edges (output of tax_annotations_from_file)."""
    _nodes = {node for edge in edges for node in edge}
    genra_nodes = {
        x.split("__")[-1].split()[0]
        for x in _nodes
        if x.startswith("g__") or len(x.split()) > 1
    }
    return genra_nodes


def get_terminals(edges: list):
    """Return leaves (terminal nodes) of a taxonomy subgraph."""
    H = nx.DiGraph()
    H.add_edges_from([x for x in edges if x[0] != x[1]])
    terminal = [x for x in H.nodes() if H.out_degree(x) == 0 and H.in_degree(x) == 1]
    terminal = [x for x in terminal if len(x.split()) == 1]
    return terminal


def get_mean(f):
    taxo_terminals = {}
    with open(f) as h:
        dct = json.load(h)

    for k, edges in dct.items():
        terminal = get_terminals(edges)
        taxo_terminals[k] = terminal
    return taxo_terminals


def genre_to_comm2vec(genres_set):
    genres_in_vocab_vecs = [
        vec[model_vocab.get(str(taxo_ids.get(x, None)), None)]
        for x in genres_set
        if model_vocab.get(str(taxo_ids.get(x, None)), None) != None
    ]
    c2v = np.mean(genres_in_vocab_vecs, axis=0)
    return c2v
    # genres_in_vocab = {taxo_ids[x] for x in genres_set if x in model_vocab.get(str(taxo_ids.get(x,None)),None) !=None }


comm2vecs_file = f"{DATA_DIR}/comm2vecs.h5"  # legacy path (kept for backwards compatibility)
comm2vecs_metadata_file = f"{DATA_DIR}/comm2vecs_metadata.tsv"


def _resolve_comm2vecs_file():
    """Return a local path to comm2vecs.h5, using cached registry copy if needed.
    Preference order: legacy packaged location -> registry cache.
    """
    if os.path.exists(comm2vecs_file):
        return comm2vecs_file
    # Use registry (auto download)
    path = model_registry.get_model_path("comm2vecs.h5", auto_download=True)
    return str(path)


def load_mgnify_c2v():
    _c2v_file = _resolve_comm2vecs_file()
    __comm2vecs = pd.read_hdf(_c2v_file, key="df")
    _comm2vecs_metadata = pd.read_csv(
        comm2vecs_metadata_file, sep="\t", index_col="SAMPLE_ID"
    )
    return __comm2vecs, _comm2vecs_metadata

