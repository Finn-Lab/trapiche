"""Biome2Vec utilities to embed taxonomy into community vectors.

Load pre-trained embeddings and aggregate taxonomy subgraphs into fixed-size
vectors. Heavy assets are accessed lazily via the Hugging Face Hub.
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
from .utils import _get_hf_model_path, load_biome_herarchy_dict, tax_annotations_from_file

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


def _resolve_model_params(model_name: str | None, model_version: str | None) -> tuple[str, str]:
    """Resolve model name/version; require explicit values to avoid hidden config accesses.

    Note: Pass model parameters from API/CLI by constructing TaxonomyToVectorParams there.
    """
    if model_name is None or model_version is None:
        raise ValueError("model_name and model_version must be provided; pass them from API/CLI using TaxonomyToVectorParams().")
    return model_name, model_version


@lru_cache
def load_biome_embeddings(load_full_model: bool = False, *, model_name: str | None = None, model_version: str | None = None) -> BiomeEmbeddings:
    """Lazy load biome embedding assets with caching.

    Parameters
    ----------
    load_full_model : bool
        If True load the full Word2Vec model (slower, more RAM). Otherwise only load KeyedVectors
        (lighter) when/if required.
    """
    # Resolve model params (explicit overrides > config defaults).
    model_name, model_version = _resolve_model_params(model_name, model_version)

    # Resolve expected files from the model repository. The file patterns
    # should match the files stored in the HuggingFace model repo. Adjust
    # patterns if your repo uses different names.
    taxonomy_vectorization_model_path = _get_hf_model_path(model_name, model_version, "community2vec_model_v*.model")
    model_vocab_file = _get_hf_model_path(model_name, model_version, "community2vec_model_vocab_v*.json")
    vec_file = _get_hf_model_path(model_name, model_version, "community2vec_model_v*.wv.vectors.npy")

    missing = [p for p in (taxonomy_vectorization_model_path, model_vocab_file, vec_file) if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing biome2vec model files: " + ", ".join(map(str, missing)) +
            f" (HF model: {model_name} version {model_version}).\nPlease ensure the model files are available in the HuggingFace model repository."
        )

    logger.info("Loading biome embeddings (full_model=%s)", load_full_model)
    # Load vocab mapping
    with open(model_vocab_file) as h:
        _model_vocab = json.load(h)
    # Memory-map vectors (fast, minimal RAM upfront)
    _vectors = np.load(vec_file, mmap_mode='r')
    # Load taxonomy ids (cached)
    _taxo_ids = load_taxonomy_ids(model_name=model_name, model_version=model_version)
    _keyed: KeyedVectors | None = None
    if load_full_model:
        # Load full model only when explicitly requested
        _keyed = Word2Vec.load(taxonomy_vectorization_model_path).wv
    return BiomeEmbeddings(keyed=_keyed, model_vocab=_model_vocab, vectors=_vectors, taxo_ids=_taxo_ids)


def load_biome2vec(load_full_model: bool = True, *, model_name: str | None = None, model_version: str | None = None) -> KeyedVectors:
    """Backward-compatible loader returning keyed vectors.

    Parameters
    ----------
    load_full_model : bool
        Whether to force loading the full model (same behaviour as before). Provided for API stability.
    """
    emb = load_biome_embeddings(load_full_model=True if load_full_model else False, model_name=model_name, model_version=model_version)
    # If user didn't request full but we didn't load, ensure keyed is available lazily.
    if emb.keyed is None:
        # Reload with full model (will overwrite cache). Simplicity > micro-optimisation.
        load_biome_embeddings.cache_clear()  # type: ignore[attr-defined]
        emb = load_biome_embeddings(load_full_model=True, model_name=model_name, model_version=model_version)
    return emb.keyed  # type: ignore[return-value]

@lru_cache
def load_taxonomy_ids(*, model_name: str | None = None, model_version: str | None = None) -> dict:
    """Load taxonomy graph and build id mapping (cached)."""
    # Resolve taxonomy graph from the HF model repo used for taxonomy vectorization.
    model_name, model_version = _resolve_model_params(model_name, model_version)
    p = _get_hf_model_path(model_name, model_version, "taxonomy_graph_*.graphml")
    if not Path(p).exists():
        raise FileNotFoundError(f"taxonomy_graph file not found: {p} (HF model: {model_name} version {model_version})")
    logger.info("Loading taxonomy_graph file=%s", p)
    taxonomy_graph = nx.read_graphml(p, node_type=str)
    taxo_ids = {x: ix for ix, x in enumerate(taxonomy_graph.nodes)}
    logger.debug("taxo_ids size=%s", len(taxo_ids))
    return taxo_ids


def get_model_vocab(*, model_name: str | None = None, model_version: str | None = None) -> dict:
    """Access the model vocabulary mapping lazily."""
    return load_biome_embeddings(model_name=model_name, model_version=model_version).model_vocab


def get_vectors(*, model_name: str | None = None, model_version: str | None = None) -> np.ndarray:
    """Access the embedding vectors lazily (memory-mapped)."""
    return load_biome_embeddings(model_name=model_name, model_version=model_version).vectors




def sentence_vectorization(terminals, *, model_name: str | None = None, model_version: str | None = None):
    """Compute mean vector from terminal nodes of a subgraph.

    Args:
        terminals: Iterable of terminal node labels.

    Returns:
        np.ndarray: Mean embedding vector.
    """
    emb = load_biome_embeddings(model_name=model_name, model_version=model_version)
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
    """Read a JSON subgraph file and return terminal nodes per key.

    Args:
        f: Path to JSON mapping sample keys to edge lists.

    Returns:
        dict: Map from key to list of terminal nodes.
    """
    taxo_terminals = {}
    with open(f) as h:
        dct = json.load(h)

    for k, edges in dct.items():
        terminal = get_terminals(edges)
        taxo_terminals[k] = terminal
    return taxo_terminals


def genre_to_taxonomy_vectorization(genres_set, *, model_name: str | None = None, model_version: str | None = None):
    """Compute mean vector for a set of genera.

    Args:
        genres_set: Set of genus names.

    Returns:
        np.ndarray: Mean vector or empty array when none map to the model.
    """
    emb = load_biome_embeddings(model_name=model_name, model_version=model_version)
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
def load_mgnify_c2v(*, model_name: str | None = None, model_version: str | None = None):
    """Load MGnify sample vectors and metadata (cached).

    Assets are fetched from the configured HF model repository.
    """
    model_name, model_version = _resolve_model_params(model_name, model_version)
    _c2v_file = Path(_get_hf_model_path(model_name, model_version, "mgnify_sample_vectors_v*.h5"))
    if not _c2v_file.exists():
        raise FileNotFoundError(
            f"mgnify_sample_vectors file not found: {_c2v_file} (HF model: {model_name} version {model_version})\n"
        )
    logger.info("Loading mgnify_sample_vectors file=%s", _c2v_file)
    biome_herarchy_dct, _ = load_biome_herarchy_dict()

    __mgnify_sample_vectors = pd.read_hdf(_c2v_file, key="df")

    _mgnify_sample_vectors_metadata_file = Path(_get_hf_model_path(model_name, model_version, "mgnify_sample_vectors_metadata_v*.tsv"))
    if not _mgnify_sample_vectors_metadata_file.exists():
        raise FileNotFoundError(
            f"mgnify_sample_vectors metadata file not found: {_mgnify_sample_vectors_metadata_file} (HF model: {model_name} version {model_version})\n"
            "Please ensure the file is present in the model repository."
        )
    logger.info("Loading mgnify_sample_vectors metadata file=%s", _mgnify_sample_vectors_metadata_file)
    _mgnify_sample_vectors_metadata = pd.read_csv(
        _mgnify_sample_vectors_metadata_file, sep="\t", index_col="SAMPLE_ID"
    )
    _mgnify_sample_vectors_metadata["BIOME_AMEND"] = _mgnify_sample_vectors_metadata.LINEAGE.map(lambda x: biome_herarchy_dct.get(x, x))
    return __mgnify_sample_vectors, _mgnify_sample_vectors_metadata

def vectorise_sample(list_of_tax_files, *, model_name: str | None = None, model_version: str | None = None):
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
    samples_vecs = {k: genre_to_taxonomy_vectorization(gs, model_name=model_name, model_version=model_version) for k, gs in samples_genus.items() if gs}

    if not samples_vecs:
        # No sample produced a vector -> shape (n_samples, 0)
        return np.zeros((n_samples, 0))

    first_vec = next(iter(samples_vecs.values()))
    vec_len = len(first_vec)

    ordered = [samples_vecs.get(i, np.zeros(vec_len)) for i in range(n_samples)]
    return np.stack(ordered, axis=0)