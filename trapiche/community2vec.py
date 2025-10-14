"""Biome2Vec embedding utilities.

Load pre-trained word2vec style embeddings and provide helpers to aggregate
taxonomy subgraphs into community vectors.
"""
from __future__ import annotations


from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd


import networkx as nx
from gensim.models import Word2Vec, KeyedVectors
from .utils import get_path, tax_annotations_from_file

import logging
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BiomeEmbeddings:
    """Container for biome2vec resources.

    Attributes
    ----------
    keyed : KeyedVectors | None
        The keyed vectors (if full model loaded) or None.
    model_vocab : dict
        Mapping from taxonomy numeric ID (as string) to vector index.
    vectors : np.ndarray
        Numpy array (memory mapped) with embedding vectors.
    taxo_ids : dict
        Mapping from taxonomy node label to numeric taxonomy ID used in model_vocab.
    """
    keyed: KeyedVectors | None
    model_vocab: dict
    vectors: np.ndarray
    taxo_ids: dict


@lru_cache
def load_biome_embeddings(load_full_model: bool = False) -> BiomeEmbeddings:
    """Lazy load biome embedding assets with caching.

    Parameters
    ----------
    load_full_model : bool
        If True load the full Word2Vec model (slower, more RAM). Otherwise only load KeyedVectors
        (lighter) when/if required.
    """
    community2vec_model_path = get_path('models/biome/community2vec/1.0/community2vec_model_v1.0.model')
    model_vocab_file = get_path('models/biome/community2vec/1.0/community2vec_model_vocab_v1.0.json')
    vec_file = get_path('models/biome/community2vec/1.0/community2vec_model_vocab_v1.0.json')

    missing = [p for p in (community2vec_model_path, model_vocab_file, vec_file) if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing biome2vec model files: " + ", ".join(map(str, missing)) +
            f" (base dir: {get_path('')}).\nPlease run `trapiche-download-models` to download the required models."
        )

    logger.info("Loading biome embeddings (full_model=%s)", load_full_model)
    # Load vocab mapping
    with open(model_vocab_file) as h:
        _model_vocab = json.load(h)
    # Memory-map vectors (fast, minimal RAM upfront)
    _vectors = np.load(vec_file, mmap_mode='r')
    # Load taxonomy ids (cached)
    _taxo_ids = load_taxonomy_ids()
    _keyed: KeyedVectors | None = None
    if load_full_model:
        # Load full model only when explicitly requested
        _keyed = Word2Vec.load(community2vec_model_path).wv
    return BiomeEmbeddings(keyed=_keyed, model_vocab=_model_vocab, vectors=_vectors, taxo_ids=_taxo_ids)


def load_biome2vec(load_full_model: bool = True) -> KeyedVectors:
    """Backward-compatible loader returning keyed vectors (full model subset).

    Parameters
    ----------
    load_full_model : bool
        Whether to force loading the full model (same behaviour as before). Provided for API stability.
    """
    emb = load_biome_embeddings(load_full_model=True if load_full_model else False)
    # If user didn't request full but we didn't load, ensure keyed is available lazily.
    if emb.keyed is None:
        # Reload with full model (will overwrite cache). Simplicity > micro-optimisation.
        load_biome_embeddings.cache_clear()  # type: ignore[attr-defined]
        emb = load_biome_embeddings(load_full_model=True)
    return emb.keyed  # type: ignore[return-value]

@lru_cache
def load_taxonomy_ids() -> dict:
    """Lazy cached load of taxonomy_graph and taxo_ids dictionary."""
    p = get_path("resources/taxonomy/taxonomy_graph.graphml")
    if not p.exists():
        raise FileNotFoundError(f"taxonomy_graph file not found: {p}")
    logger.info("Loading taxonomy_graph file=%s", p)
    taxonomy_graph = nx.read_graphml(p, node_type=str)
    taxo_ids = {x: ix for ix, x in enumerate(taxonomy_graph.nodes)}
    logger.debug("taxo_ids size=%s", len(taxo_ids))
    return taxo_ids


def get_model_vocab() -> dict:
    """Access the model vocabulary mapping lazily."""
    return load_biome_embeddings().model_vocab


def get_vectors() -> np.ndarray:
    """Access the embedding vectors lazily (memory-mapped)."""
    return load_biome_embeddings().vectors




def sentence_vectorization(terminals):
    """Compute mean vector from terminal nodes of a subgraph (community)."""
    # tax ='Laterosporus'
    emb = load_biome_embeddings()
    tix_ = [emb.taxo_ids.get(tax) for tax in terminals]
    tixs = [x for x in tix_ if x is not None]
    v_ixs_ = [emb.model_vocab.get(str(tix)) for tix in tixs]
    v_ixs = [x for x in v_ixs_ if x is not None]
    _t = [emb.vectors[v_ix] for v_ix in v_ixs]

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
    emb = load_biome_embeddings()
    _vectors = []
    for x in genres_set:
        tax_id = emb.taxo_ids.get(x)
        if tax_id is None:
            continue
        mv_ix = emb.model_vocab.get(str(tax_id))
        if mv_ix is None:
            continue
        _vectors.append(emb.vectors[mv_ix])
    if not _vectors:
        return np.zeros((0,), dtype=float)
    return np.mean(_vectors, axis=0)


@lru_cache
def load_mgnify_c2v():
    _c2v_file = get_path('models/biome/mgnify_sample_vectors/1.0/mgnify_sample_vectors_v1.0.h5')
    if not _c2v_file.exists():
        # raise error and recomend to use trapiche-download-models
        raise FileNotFoundError(
            f"mgnify_sample_vectors file not found: {_c2v_file} (base dir: {get_path('')})\n"
            "Please run `trapiche-download-models` to download the required models."
            )
    logger.info("Loading mgnify_sample_vectors file=%s", _c2v_file)
    __mgnify_sample_vectors = pd.read_hdf(_c2v_file, key="df")

    _mgnify_sample_vectors_metadata_file = get_path('models/biome/mgnify_sample_vectors/1.0/mgnify_sample_vectors_metadata_v1.0.tsv')
    if not _mgnify_sample_vectors_metadata_file.exists():
        raise FileNotFoundError(
            f"mgnify_sample_vectors metadata file not found: {_mgnify_sample_vectors_metadata_file} (base dir: {get_path('')})\n"
            "Please run `trapiche-download-models` to download the required models."
            )
    logger.info("Loading mgnify_sample_vectors metadata file=%s", _mgnify_sample_vectors_metadata_file)
    _mgnify_sample_vectors_metadata = pd.read_csv(
        _mgnify_sample_vectors_metadata_file, sep="\t", index_col="SAMPLE_ID"
    )
    return __mgnify_sample_vectors, _mgnify_sample_vectors_metadata

def vectorise_sample(list_of_tax_files):
    """Vectorise one or many samples from taxonomy annotation files.

    Flexible input forms are accepted (all are normalised internally to the
    original list-of-lists of file paths interface):

    1. Single file path (str / Path): that file represents one sample.
    2. Directory path: every *.tsv or *.tsv.gz file inside (non-recursive)
       is treated as belonging to one sample.
    3. Flat list of file paths: all those files together are one sample.
    4. List of lists of file paths (original behaviour): each inner list is a sample.

    Returns
    -------
    np.ndarray
        Array with shape (n_samples, embedding_dim). If no vectors can be
        produced an array with shape (n_samples, 0) is returned (or (0, 0) if
        there are no samples at all).
    """

    from pathlib import Path

    def _normalise(sources):  # -> list[list[str]]
        """Normalise user input to list-of-lists of file paths.

        See function docstring for accepted forms.
        """
        if isinstance(sources, (str, os.PathLike)):
            p = Path(sources)
            if p.is_dir():
                files = sorted([
                    str(f) for f in p.iterdir()
                    if f.is_file() and (f.suffix in {'.tsv', '.gz'} or f.name.endswith('.tsv.gz'))
                ])
                return [files] if files else [[]]
            elif p.is_file():
                return [[str(p)]]
            else:
                raise FileNotFoundError(f"Path not found: {p}")
        if isinstance(sources, list):
            if not sources:
                return []
            # Detect list-of-lists (or other iterables) vs flat list
            if all(isinstance(x, (list, tuple, set)) for x in sources):
                return [[str(f) for f in sample] for sample in sources]
            # Flat list of paths
            if all(isinstance(x, (str, os.PathLike)) for x in sources):
                return [[str(f) for f in sources]]
        raise TypeError(
            "Unsupported input for vectorise_run. Provide a path, a list of paths, or a list of list of paths."
        )

    samples_files = _normalise(list_of_tax_files)

    n_samples = len(samples_files)
    if n_samples == 0:
        return np.zeros((0, 0))

    samples_annots: dict[int, list] = {}
    for ix, samp_files in enumerate(samples_files):
        if not samp_files:  # skip empty lists (retain index for shape)
            continue
        for f in samp_files:
            try:
                d = tax_annotations_from_file(f)
            except Exception as e:  # defensive: keep other files processing
                print(f"Failed to parse taxonomy file {f}: {e}")
                d = None
            if d:
                samples_annots.setdefault(ix, []).extend(d)

    # Derive genus sets and vectors per sample
    samples_genus = {k: genus_from_edges_subgraph(e) for k, e in samples_annots.items()}
    samples_vecs = {k: genre_to_comm2vec(gs) for k, gs in samples_genus.items() if gs}

    if not samples_vecs:
        # No sample produced a vector -> shape (n_samples, 0)
        return np.zeros((n_samples, 0))

    first_vec = next(iter(samples_vecs.values()))
    vec_len = len(first_vec)

    ordered = [samples_vecs.get(i, np.zeros(vec_len)) for i in range(n_samples)]
    return np.stack(ordered, axis=0)