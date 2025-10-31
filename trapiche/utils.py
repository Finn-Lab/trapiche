"""General utility functions for Trapiche.

This module consolidates parsing helpers, similarity metrics, ontology
information-theoretic metrics, coloring utilities, dataset splitting,
and lightweight web / XML helpers.
"""

__all__ = ['parse_diamond', 'sanity_check_otus_annot_file', 'parse_otus_count', 'sanity_check_diamond_annot_file',
           'cosine_similarity', 'cosine_similarity_pairwise', 'jaccard_similarity', 'find_common_lineage',
           'jsonCompressed', 'split_tt', 'three_split', 'subsamp', 'longest_matching_string',
           'match_metrics', 'build_bayesian_onto', 'ia_v', 'i_T', 'roc_preds', 'semantic_distance',
           'info_theoretic_metrics', 'fbeta_score', 'set_info_theoretic_metrics', 'fetch_data_from_ebi',
           'get_project_text_description']

from functools import lru_cache
import glob
import gzip
import json
import logging
import math
import os
import re

import os
import re
import gzip
from pathlib import Path
from contextlib import contextmanager
import io

import requests
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download

import os

import logging
logger = logging.getLogger(__name__)

# --- Path helpers ---

def _get_hf_model_path(model_name: str, model_version: str, file_pattern: str) -> Path:
    """Resolve a versioned file path from a Hugging Face model repo.

    The star in file_pattern is replaced with model_version, and the file is
    downloaded to the local HF cache if needed.

    Args:
        model_name: HF repository id.
        model_version: Semantic version or tag.
        file_pattern: Pattern with a single '*' placeholder.

    Returns:
        Path: Local path to the resolved file.

    Raises:
        FileNotFoundError: If the asset cannot be resolved or downloaded.
    """
    try:
        filename = f"{model_version}/{file_pattern.replace('*', model_version)}"
        file_path = hf_hub_download(repo_id=model_name, filename=filename, repo_type="model")
        return Path(file_path)
    except Exception as e:
        raise FileNotFoundError(f"Could not find model file for {model_name} version {model_version} with pattern {file_pattern}: {e}")
# --- ---

@contextmanager
def _open_text_auto(path: str | os.PathLike, mode: str = "rt", encoding: str = "utf-8"):
    """Open plain or gzip-compressed text transparently.

    Chooses gzip when the filename ends with .gz or the file starts with gzip
    magic bytes. Falls back to plain open otherwise.
    """
    p = Path(path)
    use_gzip = p.suffix == ".gz"
    if not use_gzip:
        # Sniff first two bytes for gzip magic number 1f 8b
        try:
            with open(p, "rb") as _rb:
                magic = _rb.read(2)
            if magic == b"\x1f\x8b":
                use_gzip = True
        except OSError:
            pass
    opener = gzip.open if use_gzip else open
    # Open the file (may be binary or text depending on mode and opener)
    with opener(p, mode, encoding=encoding) as handle:  # type: ignore[arg-type]
        # If the underlying handle is in binary mode, wrap it so iteration
        # yields str objects consistently. Some callers expect text lines
        # and will pass a string separator to split(), so ensure we return
        # a text-mode file-like object.
        handle_mode = getattr(handle, "mode", mode)
        if "b" in handle_mode:
            wrapper = io.TextIOWrapper(handle, encoding=encoding)
            try:
                yield wrapper
            finally:
                # Close the wrapper to flush and detach properly. Underlying
                # handle will be closed by the outer context manager as well.
                try:
                    wrapper.close()
                except Exception:
                    pass
        else:
            yield handle


def diamond_read(f):
    """Extract taxonomy edges from a DIAMOND annotation file.

    Supports .tsv and .tsv.gz files. Uses column 15 (0-based 14) for lineage
    and builds simple genus->species-like edges.
    """
    with _open_text_auto(f, "rt", encoding="utf-8") as h:
        _diamonds_set = set()
        for raw in h:
            if not raw:
                continue
            # Normalize to str in case a binary iterator slipped through
            if isinstance(raw, (bytes, bytearray)):
                try:
                    line = raw.decode("utf-8")
                except Exception:
                    # Skip lines that can't be decoded
                    continue
            elif isinstance(raw, memoryview):
                try:
                    line = bytes(raw).decode("utf-8")
                except Exception:
                    continue
            else:
                line = raw
            # Ensure final safety: coerce non-str to str
            if not isinstance(line, str):
                try:
                    line = str(line)
                except Exception:
                    continue
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) <= 14:
                # Malformed line; skip rather than crashing
                continue
            tax = parts[14].split("=")[-1].replace("Candidatus ", "")
            _diamonds_set.add(tax)
        _diamonds = list(_diamonds_set)
    edges = set()
    for s in _diamonds:
        spl = s.split()
        if not spl:
            continue
        first_token = str(spl[0])
        if first_token and first_token[0].isalpha() and first_token[0].isupper():
                edge = (
                    spl[0],
                    s if len(spl) > 1 else "",  # pseudo-graph in diamond, connect gr with sp
                )
                edges.add(edge)
    return list(edges)


def krona_read(content):
    """Parse Krona-style taxonomy content into edge lists.

    Args:
        content: Iterable of lines (open file or list of str).

    Returns:
        list[tuple[str, str]]: Directed edges in the taxonomy graph.
    """
    edges = set()  # Use a set to avoid duplicate edges
    for _line in content:
        if _line.startswith("#") or not _line.strip():
            continue
        line = _line.replace("Candidatus ", "")
        # Split line and filter out any empty strings or strings representing empty nodes like 'k__'
        lineage = [bit for bit in re.split("[\t;]",line) if "__" in bit and not bit.endswith("__")]

        # Initialize previous valid item variable
        prev = None
        for item in lineage:
            # Skip empty taxonomy levels
            if item.endswith("__"):
                continue
            if prev is not None:
                # Create an edge between the previous valid item and the current item
                prev1, prev2 = prev.split("__")
                item1, item2 = item.split("__")
                edges.add(
                    (
                        prev1
                        + "__"
                        + prev2.split("__")[-1]
                        .replace("_", " ")
                        .replace("Candidatus ", ""),
                        item1
                        + "__"
                        + item2.split("__")[-1]
                        .replace("_", " ")
                        .replace("Candidatus ", ""),
                    )
                )
            prev = item
    return list(edges)


def tax_annotations_from_file(f):
    """Extract taxonomy annotations from an input file.

    DIAMOND outputs are detected by filename. Otherwise a Krona-style TSV is
    assumed. Gzip files are supported.

    Args:
        f: File path.

    Returns:
        list[tuple[str, str]] | None: Edge list or None if parsing fails.
    """
    d = None
    logger.debug(f"tax_annotations_from_file called file={f}")
    if "diamond" in f:
        try:
            d = diamond_read(f)
        except Exception as e:
            logger.error(f"diamond_read failed file={f} error={e}")
    else:
        try:
            with open(f) as content:
                d = krona_read(content)
        except Exception as e:
            logger.warning(f"open krona failed file={f} error={e}")
            try:
                with gzip.open(f, "rt") as content:  # 'rt' mode for reading text
                    d = krona_read(content)
            except Exception as e2:
                logger.error(f"gzip krona failed file={f} error={e2}")
    logger.debug(f"tax_annotations_from_file result n_edges={len(d) if d is not None else 0}")
    return d


# --- ---

@lru_cache
def load_biome_herarchy_dict():
    """Load amended biome hierarchy mapping from HF assets (cached)."""
    from .config import TaxonomyToVectorParams as _T2V
    from .utils import _get_hf_model_path
    _p = _T2V()
    p = _get_hf_model_path(_p.hf_model, _p.model_version, "biome_herarchy_amended_*.json")

    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    logger.debug(f"Loading biome_herarchy_dct from file={p}")
    # Load the biome hierarchy dictionary from the compressed JSON file
    with open(p, encoding="utf-8") as f:
        biome_herarchy_dct = json.load(f)
    logger.debug(f"biome_herarchy_dct loaded n_keys={len(biome_herarchy_dct) if biome_herarchy_dct else 0}")
    biome_herarchy_dct_reversed = {v: k for k, v in biome_herarchy_dct.items()}
    return biome_herarchy_dct, biome_herarchy_dct_reversed


def parse_diamond(
    content: list,  # diamond_taxo_annotation content. Usually is the result of list(open(PATH_FILE).read())
):
    """Extract unique taxa from DIAMOND lines.

    Args:
        content: Lines from a DIAMOND TSV file.

    Returns:
        set[str]: Unique taxon strings.
    """
    mix = {
        line.split("\t")[14].split("=")[-1].replace("Candidatus ", "")
        for line in content[:-1]
    }
    return mix


def sanity_check_otus_annot_file(filepath):
    """Collect lines from an OTU annotation file (light validation)."""
    with open(filepath, "r") as f:
        content = list(f)
    for line in content:
        if not line.startswith("#"):
            # Future: add structural validation here
            break
    return content


def parse_otus_count(
    content: list,  # diamond_taxo_annotation content. Usually is the result of list(open(PATH_FILE).read())
):
    """Extract unique taxa from an OTU count file's content.

    Args:
        content: Lines from an OTU table.

    Returns:
        set[str]: Unique taxon tokens at the deepest level.
    """
    se = [
        " ".join(";".join(line.strip().split("\t")[1:]).replace("__", "//").split("_"))
        .replace("//", "__")
        .replace("Candidatus ", "")
        for line in content
    ]
    mix = {x.split("__")[-1] for l in se for x in l.split(";")}
    return mix


def sanity_check_diamond_annot_file(filepath):
    """Read a DIAMOND TSV (or .gz) and return its content lines.

    Placeholder for future header/format validation.
    """

    # List of valid headers
    # valid_headers = {'uniref90_ID', 'contig_name', 'percentage_of_identical_matches', 'length', 'mismatch',
    #                  'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore', 'protein_name',
    #                  'num_in_cluster', 'taxonomy', 'tax_id', 'rep_id','lenght'}
    # Open the file for reading
    # Support both compressed and uncompressed diamond outputs
    opener = gzip.open if str(filepath).endswith('.gz') else open
    with opener(filepath, "rt") as h:  # type: ignore[arg-type]
        content = list(h)

        # Check if the first line matches the valid headers
        for ix, line in enumerate(content):
            if line[0] == "#":
                continue
            # line = line.strip().split('\t')
            # if ix <1:
            #     if len(set(line)-valid_headers)!=0:
            #         return False
            # Read the rest of the lines
            else:
                # Placeholder for additional validation logic if needed
                pass
        return content


from numpy import dot
from numpy.linalg import norm


def cosine_similarity(a, b):
    """Return cosine similarity between two vectors."""
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim


def cosine_similarity_pairwise(A, B):
    """Compute pairwise cosine similarity between row vectors in A and B."""
    dot_product = np.dot(A, B.T)
    norm_a = np.linalg.norm(A, axis=1)
    norm_b = np.linalg.norm(B, axis=1)
    return dot_product / (np.outer(norm_a, norm_b))


def jaccard_similarity(set1, set2):
    """Return Jaccard similarity between two sets."""
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if len(union) == 0:
        return 0
    else:
        return len(intersection) / len(union)


def find_common_lineage(lineages):
    """Find the longest common prefix (by nodes) across lineages."""
    if not lineages:
        return ""
    if len(lineages) == 1:
        return lineages[0]

    # Split lineages into list of nodes
    lineage_nodes = [l.split(":") for l in lineages]

    # Define a common lineage list
    common_lineage = []

    # Iterate over the nodes in the first lineage
    for i, node in enumerate(lineage_nodes[0]):
        # Check if this node is present at the same position in 50% or more of the other lineages
        if (
            sum(i < len(l) and l[i] == node for l in lineage_nodes)
            >= len(lineage_nodes) / 2
        ):
            # If so, append it to the common lineage
            common_lineage.append(node)
        else:
            # If not, we've found the end of the common lineage, so break
            break

    # Return the common lineage as a string
    return ":".join(common_lineage)


class jsonCompressed:
    """Dump/read JSON objects using gzip compression."""

    def __init__(self):
        pass

    @staticmethod
    def dump(data, outfile):
        """Serialize object to gzip-compressed JSON (one object)."""
        json_str = json.dumps(data) + "\n"
        with gzip.open(outfile, "wt", encoding="utf-8") as fout:
            fout.write(json_str)

    @staticmethod
    def read(infile):
        """Read a gzip-compressed JSON object from disk."""
        with gzip.open(infile, "rt", encoding="utf-8") as fin:
            return json.loads(fin.read())



def split_tt(df, frac, rs, lin):
    """Mark test samples by leaving out complete projects.

    Adds a boolean IS_TEST column to help assess performance fairly.
    """
    test_samples = []
    if frac != 0:  # skip this step if models are trained with the full dataset
        if type(lin) == str:
            depth = len(lin.split(":"))
            tdf = df[df.max_depth > depth]
            tdf["d3"] = [":".join(x.split(":")[: depth + 1]) for x in tdf.lineage]
        else:
            tdf = df
            tdf["d3"] = lin
        # tdf['d3'] = tdf.lineage
        for g, gr in tdf.groupby("d3"):
            expected = int(gr.shape[0] * frac)
            if gr.project.nunique() == 1:
                test_samples.extend(
                    gr.sample(frac=frac, random_state=rs).SAMPLE_ID.values
                )
            else:
                accum = 0
                for ix, n in (
                    gr.project.value_counts()
                    .sample(frac=1, random_state=rs)
                    .iteritems()
                ):
                    momo = tdf[tdf.project == ix].SAMPLE_ID
                    test_samples.extend(momo)
                    accum += momo.shape[0]
                    if accum >= expected:
                        break
    test_ixes = df[df.SAMPLE_ID.isin(test_samples)].index
    df["IS_TEST"] = [False] * df.shape[0]
    df.loc[test_ixes, ("IS_TEST")] = True


def three_split(df, random_state=None, frac_=0.2):
    """Split a dataframe into train/val/test by non-overlapping projects."""
    gr = pd.DataFrame(df)
    ann = lambda x: re.sub(":*$", "", ":".join((x.split(":") + [""] * 3)[:3]))
    gr["top"] = gr.lineage.map(ann)
    if gr.lineage.nunique() < 1:
        print("NOTHING\n")
        # continue
    split_tt(gr, frac_, random_state, gr["top"])
    test_df = gr[gr.IS_TEST == True]
    train_df = gr[gr.IS_TEST == False]
    split_tt(train_df, frac_, random_state, gr["top"])
    train_df.IS_TEST.value_counts()
    test_df.IS_TEST.value_counts()
    val_df = train_df[train_df.IS_TEST == True]
    train_df = train_df[train_df.IS_TEST == False]
    return train_df, val_df, test_df


def subsamp(df, random_state=None, top=1200):
    """Subsample to keep at most K samples per biome label."""
    samps = []
    for g, gr in df.groupby("lineage"):
        if gr.shape[0] > top:
            samps.extend(gr.sample(top, random_state=random_state).SAMPLE_ID.values)
        else:
            samps.extend(gr.SAMPLE_ID.values)
    new_df = df[df.SAMPLE_ID.isin(samps)]
    return new_df


from difflib import SequenceMatcher


def longest_matching_string(string1, string2):
    """Return the longest common substring between two strings."""
    match = SequenceMatcher(None, string1, string2).find_longest_match()
    lm = string1[match.a : match.a + match.size]
    return lm


def match_metrics(_gt, _pred):
    """Compute simple precision/recall between two lineage strings."""
    # remove the root node
    gt, pred = [set(x.replace("root:", "").split(":")) for x in [_gt, _pred]]
    recall = len(gt & pred) / len(gt)
    precision = len(gt & pred) / len(pred)
    return recall, precision


import networkx as nx


"""Bayesian network of GOLD."""


def build_bayesian_onto(onto_net):
    """Construct a Bayesian network with simple conditional probabilities."""
    nodes = list(onto_net.nodes)
    for node in nodes:
        if node == "root":
            onto_net.nodes[node]["conditional_probability"] = 1
            continue
        _parent = list(onto_net.in_edges(node))[0][0]
        n_siblings = len(onto_net.out_edges(_parent))
        # print(node)
        onto_net.nodes[node]["conditional_probability"] = 1 / n_siblings


def ia_v(term, graph):
    """Return information accretion of a node in the ontology."""
    cond_prob = graph.nodes[term]["conditional_probability"]
    information_accretion = np.log2(1 / cond_prob)
    return information_accretion


def i_T(term, graph):
    """Information content of a term or set of terms.

    If term is a string, use its ancestors; if iterable, use the set itself.
    """
    if type(term) == str:
        veT = nx.ancestors(graph, term) | {term}
    else:
        veT = term
    _info_content = [ia_v(x, graph) for x in veT]
    return np.sum(_info_content)


def roc_preds(true, path, probs, threshold):
    """Return path elements with probability >= threshold (ROC helper)."""
    return path[np.where(probs >= threshold)[0]]


def semantic_distance(ru, mi, k=2):
    """Compute L-k distance given remaining uncertainty and misinformation."""
    return ((ru**k) + (mi**k)) ** (1 / k)


def info_theoretic_metrics(T, P, graph):
    """Compute precision/recall and info-theoretic metrics for a prediction."""
    # get ancestors
    Tt = nx.ancestors(graph, T) | {T}
    Pt = nx.ancestors(graph, P) | {P}
    intersection = Pt & Tt  # for recall precision

    precision = len(intersection) / len(Pt)
    recall = len(intersection) / len(Tt)

    # remaining_uncertainty(T,P): " an analog of recall in the information-theoretic framework"
    ru_nodes = Tt - Pt
    remaining_uncertainty = i_T(ru_nodes, graph)

    # misinformation(T,P): " an analog of precision in the information-theoretic framework"
    mi_nodes = Pt - Tt
    misinformation = i_T(mi_nodes, graph)

    # weighted precision
    wp_nodes_1 = Tt & Pt
    wp_nodes_2 = Pt
    w_precision = i_T(wp_nodes_1, graph) / i_T(wp_nodes_2, graph)
    if np.isnan(w_precision):  # handle NaN due to empty denominator
        w_precision = 0.0

    # weighted recall
    wr_nodes_1 = Tt & Pt
    wr_nodes_2 = Tt
    w_recall = i_T(wr_nodes_1, graph) / i_T(wr_nodes_2, graph)

    return (
        precision,
        recall,
        misinformation,
        remaining_uncertainty,
        w_precision,
        w_recall,
    )


def fbeta_score(precision, recall, beta=1):
    """Compute F-beta score with safe zero handling."""
    if (beta**2 * precision) + recall == 0:
        return 0
    fscore = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    return fscore


def set_info_theoretic_metrics(true, pred, graph, k=1):
    """Compute info-theoretic metrics for sets of true/pred terms."""
    (
        precision,
        recall,
        misinformation,
        remaining_uncertainty,
        w_precision,
        w_recall,
    ) = list(zip(*[info_theoretic_metrics(T, P, graph) for T, P in zip(true, pred)]))

    ture_info_contents = [i_T(t, graph) for t in true]

    # ru(true,pred): "calculate remaining_uncertainty for a set of terms"
    ru = pd.Series(remaining_uncertainty).sum() / len(true)

    # mi(true,pred): "calculate misinformation for a set of terms"
    mi = pd.Series(misinformation).sum() / len(true)

    # wru(true,pred): "calculate weighted remaining_uncertainty for a set of terms. Downweight shallow terms"
    wru = (
        sum([tic * _ru for tic, _ru in zip(ture_info_contents, remaining_uncertainty)])
        / pd.Series(ture_info_contents).sum()
    )

    # wmi(true,pred): "calculate weighted misinformation for a set of terms. Downweight shallow terms"
    wmi = (
        sum([tic * _mi for tic, _mi in zip(ture_info_contents, misinformation)])
        / pd.Series(ture_info_contents).sum()
    )

    sd = semantic_distance(ru, mi, k=k)
    wsd = semantic_distance(wru, wmi, k=k)

    # precision and recall for all
    pr = pd.Series(precision).sum() / len(true)
    rc = pd.Series(recall).sum() / len(true)
    f1 = fbeta_score(pr, rc)

    # wpr(true,pred): "calculate weighted precision for a set of terms. Downweight shallow terms"
    # wpr = pd.Series(w_precision).sum()  / len(true)
    wpr = pd.Series(w_precision).sum() / len(true)
    # if wpr==float('nan'):
    # wrc(true,pred): "calculate weighted recall for a set of terms. Downweight shallow terms"
    wrc = pd.Series(w_recall).sum() / len(true)

    return pr, rc, wpr, wrc, mi, ru, wmi, wru, f1, sd, wsd


def fetch_data_from_ebi(acc):
    """Fetch ENA XML metadata for an accession.

    Returns the raw XML string or None if the request fails.
    """
    url = "https://www.ebi.ac.uk/ena/browser/api/xml/{}?download=false".format(acc)
    response = requests.get(url)
    if response.status_code == 200:
        xml_content = response.text
        return xml_content
    else:
        return None


def get_project_text_description(acc):
    """Fetch project title and description from ENA; returns (title, description) or (None, None)."""
    xml_data = fetch_data_from_ebi(acc)
    if not xml_data:
        return None, None
    try:
        root = ET.fromstring(xml_data)
    except Exception:
        return None, None
    title_el = root.find(".//TITLE")
    desc_el = root.find(".//DESCRIPTION")
    title = title_el.text if title_el is not None else None
    description = desc_el.text if desc_el is not None else None
    return title, description



def get_similar_predictions(probabilities, diff_thresh=0.05, ratio_thresh=0.9):
    """
    Identify indices of predictions with probabilities similar to the top one.

    This function expects a 1-D numpy array (or array-like) of probabilities and
    returns a list of integer indices (into the original array) whose
    probabilities are within either the absolute difference threshold
    (`diff_thresh`) or the relative ratio threshold (`ratio_thresh`) compared
    to the top probability. The top (largest) probability's index is always
    included.
    
    Args:
        probabilities (np.ndarray or sequence): 1-D array-like of probabilities.
        diff_thresh (float): Max allowed absolute difference between top and others.
        ratio_thresh (float): Min allowed ratio (p_i / top_p) to count as similar.

    Returns:
        list[int]: Indices of entries similar to the top prediction, sorted by
                   decreasing probability (so indices follow the order of
                   descending probabilities).
    """
    # Convert to numpy array for consistent behavior and flatten
    probs = np.asarray(probabilities)
    # Handle scalar / 0-d arrays by treating them as length-1 arrays
    probs = probs.reshape(-1) if probs.ndim == 0 else probs.flatten()

    if probs.ndim != 1:
        raise ValueError("probabilities must be a 1-D array-like of scores")
    if probs.size == 0:
        return []

    # If all values are NaN, nothing to return
    if np.isnan(probs).all():
        return []

    # Sort indices by descending probability (stable sort)
    # use negative to avoid reversing after argsort
    sorted_idx = np.argsort(-probs, kind="stable")

    # Ensure indices are integers and safe to index
    top_idx = int(sorted_idx[0])
    top_p = float(probs[top_idx])

    similar_indices = [top_idx]

    # Iterate remaining indices in descending order
    for idx in sorted_idx[1:]:
        idx = int(idx)
        p = probs[idx]
        # Skip NaNs quietly
        if np.isnan(p):
            continue

        # Compute diff and ratio robustly
        diff = abs(top_p - float(p))
        if top_p == 0:
            # treat ratio as 1.0 only when both are zero
            ratio = 1.0 if float(p) == 0.0 else 0.0
        else:
            ratio = float(p) / top_p

        if diff <= diff_thresh or ratio >= ratio_thresh:
            similar_indices.append(idx)
        else:
            break

    return similar_indices

def obj_to_serializable(obj):
    """Recursively convert common non-JSON types to JSON-serializable Python types.

    - numpy scalars -> Python scalars via .item()
    - numpy arrays -> lists via .tolist()
    - Path -> str
    - bytes -> decoded str (fallback to repr)
    - sets/tuples -> lists
    - dict/list -> recursively converted
    - non-finite floats -> None
    - fallback: str(obj)
    """
    # simple primitives
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        # convert NaN/inf to None to keep NDJSON parsable
        if math.isfinite(obj):
            return obj
        return None

    # Paths
    if isinstance(obj, Path):
        return str(obj)

    # bytes
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8")
        except Exception:
            return repr(obj)

    # Numpy types (best-effort, numpy may not be installed)
    try:
        import numpy as _np
        if isinstance(obj, _np.generic):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    # dicts
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                key = k if isinstance(k, str) else str(k)
            except Exception:
                key = str(k)
            out[key] = obj_to_serializable(v)
        return out

    # iterables -> list
    if isinstance(obj, (list, tuple, set)):
        return [obj_to_serializable(v) for v in obj]

    # fallback to string
    try:
        return str(obj)
    except Exception:
        return None
