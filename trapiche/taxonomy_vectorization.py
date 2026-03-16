"""Biome2Vec utilities to embed taxonomy into community vectors.

Load pre-trained embeddings and aggregate taxonomy subgraphs into fixed-size
vectors. Heavy assets are accessed lazily via the Hugging Face Hub.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec

from .utils import (
    _get_hf_model_path,
    load_biome_herarchy_dict,
    normalize_to_list_of_str,
    read_taxonomy_study_tsv_cached,
    tax_annotations_from_file,
)

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
        raise ValueError(
            "model_name and model_version must be provided; pass them from API/CLI using TaxonomyToVectorParams()."
        )
    return model_name, model_version


@lru_cache
def load_biome_embeddings(
    load_full_model: bool = False,
    *,
    model_name: str | None = None,
    model_version: str | None = None,
) -> BiomeEmbeddings:
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
    taxonomy_vectorization_model_path = _get_hf_model_path(
        model_name, model_version, "community2vec_model_v*.model"
    )
    model_vocab_file = _get_hf_model_path(
        model_name, model_version, "community2vec_model_vocab_v*.json"
    )
    vec_file = _get_hf_model_path(
        model_name, model_version, "community2vec_model_v*.wv.vectors.npy"
    )

    missing = [
        p
        for p in (taxonomy_vectorization_model_path, model_vocab_file, vec_file)
        if not Path(p).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing biome2vec model files: "
            + ", ".join(map(str, missing))
            + f" (HF model: {model_name} version {model_version}).\nPlease ensure the model files are available in the HuggingFace model repository."
        )

    logger.info("Loading biome embeddings (full_model=%s)", load_full_model)
    # Load vocab mapping
    with open(model_vocab_file) as h:
        _model_vocab = json.load(h)
    # Memory-map vectors (fast, minimal RAM upfront)
    _vectors = np.load(vec_file, mmap_mode="r")
    # Load taxonomy ids (cached)
    _taxo_ids = load_taxonomy_ids(model_name=model_name, model_version=model_version)
    _keyed: KeyedVectors | None = None
    if load_full_model:
        # Load full model only when explicitly requested
        _keyed = Word2Vec.load(taxonomy_vectorization_model_path).wv
    return BiomeEmbeddings(
        keyed=_keyed, model_vocab=_model_vocab, vectors=_vectors, taxo_ids=_taxo_ids
    )


def load_biome2vec(
    load_full_model: bool = True, *, model_name: str | None = None, model_version: str | None = None
) -> KeyedVectors:
    """Backward-compatible loader returning keyed vectors.

    Parameters
    ----------
    load_full_model : bool
        Whether to force loading the full model (same behaviour as before). Provided for API stability.
    """
    emb = load_biome_embeddings(
        load_full_model=True if load_full_model else False,
        model_name=model_name,
        model_version=model_version,
    )
    # If user didn't request full but we didn't load, ensure keyed is available lazily.
    if emb.keyed is None:
        # Reload with full model (will overwrite cache). Simplicity > micro-optimisation.
        load_biome_embeddings.cache_clear()  # type: ignore[attr-defined]
        emb = load_biome_embeddings(
            load_full_model=True, model_name=model_name, model_version=model_version
        )
    return emb.keyed  # type: ignore[return-value]


@lru_cache
def load_taxonomy_ids(*, model_name: str | None = None, model_version: str | None = None) -> dict:
    """Load taxonomy graph and build id mapping (cached)."""
    # Resolve taxonomy graph from the HF model repo used for taxonomy vectorization.
    model_name, model_version = _resolve_model_params(model_name, model_version)
    p = _get_hf_model_path(model_name, model_version, "taxonomy_graph_*.graphml")
    if not Path(p).exists():
        raise FileNotFoundError(
            f"taxonomy_graph file not found: {p} (HF model: {model_name} version {model_version})"
        )
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


def sentence_vectorization(
    terminals, *, model_name: str | None = None, model_version: str | None = None
):
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
        x.split("__")[-1].split()[0] for x in _nodes if x.startswith("g__") or len(x.split()) > 1
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


def genre_to_taxonomy_vectorization(
    genres_set, *, model_name: str | None = None, model_version: str | None = None
):
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

    _mgnify_sample_vectors_metadata_file = Path(
        _get_hf_model_path(model_name, model_version, "mgnify_sample_vectors_metadata_v*.tsv")
    )
    if not _mgnify_sample_vectors_metadata_file.exists():
        raise FileNotFoundError(
            f"mgnify_sample_vectors metadata file not found: {_mgnify_sample_vectors_metadata_file} (HF model: {model_name} version {model_version})\n"
            "Please ensure the file is present in the model repository."
        )
    logger.info(
        "Loading mgnify_sample_vectors metadata file=%s", _mgnify_sample_vectors_metadata_file
    )
    _mgnify_sample_vectors_metadata = pd.read_csv(
        _mgnify_sample_vectors_metadata_file, sep="\t", index_col="sample_id"
    )
    _mgnify_sample_vectors_metadata["BIOME_AMEND"] = _mgnify_sample_vectors_metadata.LINEAGE.map(
        lambda x: biome_herarchy_dct.get(x, x)
    )
    _mgnify_sample_vectors_metadata["LINEAGE_LENGTH"] = _mgnify_sample_vectors_metadata.LINEAGE.map(
        lambda x: len(x.split(":")) if pd.notna(x) else 0
    )
    _mgnify_sample_vectors_metadata = _mgnify_sample_vectors_metadata[
        _mgnify_sample_vectors_metadata["LINEAGE_LENGTH"] >= 3
    ]
    return __mgnify_sample_vectors, _mgnify_sample_vectors_metadata


def vectorise_samples(
    samples_sequence: Sequence[dict[str, Any]],
    *,
    model_name: str | None = None,
    model_version: str | None = None,
):
    """Vectorise one or many samples from taxonomy annotation files.

    samples_sequence must have EITHER keys:
      - {"sample_taxonomy_paths": ...}  (preferred)
      - {"taxonomy_files_paths": ...}   (legacy alias)
      - {"study_taxonomy_path": ..., "sample_id": ...}

    Returns
    -------
    np.ndarray
        Array with shape (n_samples, embedding_dim). If no vectors can be
        produced an array with shape (n_samples, 0) is returned (or (0, 0) if
        there are no samples at all).
    """
    n_samples = len(samples_sequence)
    if n_samples == 0:
        return np.zeros((0, 0))

    samples_annots: dict[int, list] = {}
    for ix, sample_dict in enumerate(samples_sequence):

        sample_taxonomy_terms = None
        # check if study_taxonomy_path and sample_id are provided
        if "study_taxonomy_path" in sample_dict and "sample_id" in sample_dict:
            study_taxonomy_path = sample_dict["study_taxonomy_path"]
            sample_id = sample_dict["sample_id"]
            try:
                sample_taxonomy_terms = read_taxonomy_study_tsv_cached(study_taxonomy_path).get(
                    sample_id
                )
                if sample_taxonomy_terms is None:
                    logger.error(
                        f"Sample ID {sample_id} not found in taxonomy file {study_taxonomy_path} while study_taxonomy_path and sample_id were provided."
                    )
            except Exception as e:  # defensive: keep other files processing
                logger.error(f"Failed to parse taxonomy file {study_taxonomy_path}: {e}")
            samples_annots.setdefault(ix, []).extend(
                sample_taxonomy_terms if sample_taxonomy_terms else []
            )

        # fallback to taxonomy_paths if sample_taxonomy_terms is not found
        # support both "sample_taxonomy_paths" and legacy "taxonomy_files_paths" key
        _tax_paths_key = (
            "sample_taxonomy_paths"
            if "sample_taxonomy_paths" in sample_dict
            else "taxonomy_files_paths" if "taxonomy_files_paths" in sample_dict else None
        )
        if sample_taxonomy_terms is None and _tax_paths_key is not None:
            sample_taxonomy_paths = sample_dict[_tax_paths_key]

            # normalise to list of strings
            sample_taxonomy_paths = normalize_to_list_of_str(sample_taxonomy_paths)

            if not sample_taxonomy_paths:  # skip empty lists (retain index for shape)
                logger.warning(f"No taxonomy files provided for sample index {ix}.")
                continue
            for f in sample_taxonomy_paths:
                try:
                    _sample_taxonomy_terms = tax_annotations_from_file(f)
                except Exception as e:  # defensive: keep other files processing
                    print(f"Failed to parse taxonomy file {f}: {e}")
                    _sample_taxonomy_terms = []

            samples_annots.setdefault(ix, []).extend(
                _sample_taxonomy_terms if _sample_taxonomy_terms else []
            )

    # Derive genus sets and vectors per sample
    samples_genus = {k: genus_from_edges_subgraph(e) for k, e in samples_annots.items()}
    samples_vecs = {
        k: genre_to_taxonomy_vectorization(gs, model_name=model_name, model_version=model_version)
        for k, gs in samples_genus.items()
        if gs
    }

    # Some samples may yield empty vectors (e.g., genera not present in the model).
    # Only consider non-empty vectors to determine the embedding dimensionality.
    non_empty = {k: v for k, v in samples_vecs.items() if isinstance(v, np.ndarray) and v.size > 0}

    if not non_empty:
        # No sample produced a non-empty vector -> shape (n_samples, 0)
        return np.zeros((n_samples, 0))

    # Embedding dimension inferred from any non-empty vector (all should match model dim)
    vec_len = len(next(iter(non_empty.values())))

    # Build ordered list aligning to all input samples; use zero vector when empty/missing
    ordered = []
    for i in range(n_samples):
        v = samples_vecs.get(i)
        if v is None or v.size == 0:
            ordered.append(np.zeros((vec_len,), dtype=float))
        else:
            # Defensive: if a mismatched shape appears, coerce to expected with pad/trunc
            if len(v) != vec_len:
                logger.warning(
                    "Inconsistent vector length for sample index %s: got %s, expected %s; coercing",
                    i,
                    len(v),
                    vec_len,
                )
                if len(v) > vec_len:
                    v = v[:vec_len]
                else:
                    v = np.pad(v, (0, vec_len - len(v)), mode="constant")
            ordered.append(v.astype(float, copy=False))

    return np.stack(ordered, axis=0)
