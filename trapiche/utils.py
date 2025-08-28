"""General utility functions for Trapiche.

This module consolidates parsing helpers, similarity metrics, ontology
information-theoretic metrics, coloring utilities, dataset splitting,
and lightweight web / XML helpers.
"""

__all__ = ['parse_diamond', 'sanity_check_otus_annot_file', 'parse_otus_count', 'sanity_check_diamond_annot_file',
           'cosine_similarity', 'cosine_similarity_pairwise', 'jaccard_similarity', 'find_common_lineage',
           'jsonCompressed', 'rand_cmap', 'split_tt', 'three_split', 'subsamp', 'longest_matching_string',
           'match_metrics', 'build_bayesian_onto', 'ia_v', 'i_T', 'roc_preds', 'semantic_distance',
           'info_theoretic_metrics', 'fbeta_score', 'set_info_theoretic_metrics', 'fetch_data_from_ebi',
           'get_project_text_description']

from functools import lru_cache
import glob
import gzip
import json
import logging
import os
import re

import os
import re
import gzip
from pathlib import Path
from contextlib import contextmanager

import requests
import xml.etree.ElementTree as ET
import numpy as np
from collections import Counter
import pandas as pd
from pathlib import Path
from platformdirs import user_cache_dir
import os

import logging
logger = logging.getLogger(__name__)

# --- Path helpers ---

APP_NAME = "trapiche"

def get_cache_dir() -> Path:
    env = os.getenv("TRAPICHE_CACHE")
    if env:
        path = Path(env).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    path = Path(user_cache_dir("trapiche"))
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_path(relative: str) -> Path:
    """Join a path under the base data dir (no existence check)."""
    return get_cache_dir() / relative

# --- ---

@contextmanager
def _open_text_auto(path: str | os.PathLike, mode: str = "rt", encoding: str = "utf-8"):
    """Open plain text or gzip-compressed text transparently.

    Chooses gzip when the filename ends with .gz or the file starts with the
    gzip magic bytes. Falls back to plain open otherwise.
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
    with opener(p, mode, encoding=encoding) as handle:  # type: ignore[arg-type]
        yield handle


def diamond_read(f):
    """Extract taxonomy edges from a diamond functional annotation file (.tsv or .tsv.gz).

    The diamond annotation schema places the taxonomy lineage (last taxon) in
    column 15 (0-based index 14). We capture unique taxon strings, then build
    simple genus->species style edges (capitalised first token -> full string).
    """
    with _open_text_auto(f, "rt", encoding="utf-8") as h:
        try:
            _diamonds = list({
                line.split("\t")[14].split("=")[-1].replace("Candidatus ", "")
                for line in h
                if line and not line.startswith("#")
            })
        except IndexError as e:
            raise ValueError(f"Diamond file appears malformed (missing expected columns): {f}") from e
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
    """function to read mseq.txt files"""
    edges = set()  # Use a set to avoid duplicate edges
    for _line in content:
        line = _line.replace("Candidatus ", "")
        # Split line and filter out any empty strings or strings representing empty nodes like 'k__'
        parts = [
            part
            for part in line.strip().split("\t")
            if part and not part.endswith("__")
        ]
        # Skip the count at the beginning of each line
        lineage = parts[1:]

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
    """Function to extract taxo_annots from file"""
    d = None
    logger.info(f"tax_annotations_from_file called file={f}")
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
    logger.info(f"tax_annotations_from_file result n_edges={len(d) if d is not None else 0}")
    return d


# --- ---

@lru_cache
def load_biome_herarchy_dict():
    p = get_path("resources/biome/biome_herarchy_amended.json.gz")
    if not p.exists():
        raise FileNotFoundError(f"biome_herarchy_amended.json.gz not found at {p}")
    logger.debug(f"Loading biome_herarchy_dct from file={p}")
    biome_herarchy_dct = jsonCompressed.read(p)
    logger.debug(f"biome_herarchy_dct loaded n_keys={len(biome_herarchy_dct) if biome_herarchy_dct else 0}")
    return biome_herarchy_dct

def parse_diamond(
    content: list,  # diamond_taxo_annotation content. Usually is the result of list(open(PATH_FILE).read())
):
    """
    Given a diamond_taxo_annotation content (in for of a python list), extract the taxonomic annotations in it
    """
    mix = {
        line.split("\t")[14].split("=")[-1].replace("Candidatus ", "")
        for line in content[:-1]
    }
    return mix


def sanity_check_otus_annot_file(filepath):
    """Lightweight validation + content collection for OTU annotation file.

    Returns the list of lines (content) regardless; placeholder for future validation.
    """
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
    """
    Given a OTU conut content (in for of a python list), extract the taxonomic annotations in it
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
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim


def cosine_similarity_pairwise(A, B):
    dot_product = np.dot(A, B.T)
    norm_a = np.linalg.norm(A, axis=1)
    norm_b = np.linalg.norm(B, axis=1)
    return dot_product / (np.outer(norm_a, norm_b))


def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if len(union) == 0:
        return 0
    else:
        return len(intersection) / len(union)


def find_common_lineage(lineages):
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
    "dump and read data objects into a compressed json"

    def __init__(self):
        pass

    @staticmethod
    def dump(data, outfile):
        """Serialize data object to gzip-compressed JSON (one object)."""
        json_str = json.dumps(data) + "\n"
        with gzip.open(outfile, "wt", encoding="utf-8") as fout:
            fout.write(json_str)

    @staticmethod
    def read(infile):
        with gzip.open(infile, "rt", encoding="utf-8") as fin:
            return json.loads(fin.read())


# Generate random colormap
def rand_cmap(
    nlabels, type="bright", first_color_black=True, last_color_black=False, verbose=True
):
    """
    From https://github.com/delestro/rand_cmap

    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    import colorsys

    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    if type not in ("bright", "soft"):
        print('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print("Number of labels: " + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == "bright":
        randHSVcolors = [
            (
                np.random.uniform(low=0.0, high=1),
                np.random.uniform(low=0.2, high=1),
                np.random.uniform(low=0.9, high=1),
            )
            for i in range(nlabels)
        ]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(
                colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2])
            )

        if first_color_black:
            randRGBcolors[0] = (0.0, 0.0, 0.0)
        if last_color_black:
            randRGBcolors[-1] = (0.0, 0.0, 0.0)

        random_colormap = LinearSegmentedColormap.from_list(
            "new_map", randRGBcolors, N=nlabels
        )

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == "soft":
        low = 0.6
        high = 0.95
        randRGBcolors = [
            (
                np.random.uniform(low=low, high=high),
                np.random.uniform(low=low, high=high),
                np.random.uniform(low=low, high=high),
            )
            for i in range(nlabels)
        ]

        if first_color_black:
            randRGBcolors[0] = (0.0, 0.0, 0.0)
        if last_color_black:
            randRGBcolors[-1] = (0.0, 0.0, 0.0)
        random_colormap = LinearSegmentedColormap.from_list(
            "new_map", randRGBcolors, N=nlabels
        )

    # Display colorbar
    if verbose:
        from matplotlib import colorbar, colors
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(
            ax,
            cmap=random_colormap,
            norm=norm,
            spacing="proportional",
            ticks=None,
            boundaries=bounds,
            format="%1i",
            orientation="horizontal",
        )

    return random_colormap


def split_tt(df, frac, rs, lin):
    "add a column to dataframe marking test samples, based on leaving out complete projects to better asses performance"
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
    "split a dataframe in tr,te,val using non,overlaping projects"
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
    "subsample a dataframe to keep biome with less than K samples per biome"
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
    """
    find longest matching string
    """
    match = SequenceMatcher(None, string1, string2).find_longest_match()
    lm = string1[match.a : match.a + match.size]
    return lm


def match_metrics(_gt, _pred):
    """
    given two lineages return the leng of the max_matching string and the distance between biome nodes. Nedagtive means that is not an extention of the ground truth
    """
    # remove the root node
    gt, pred = [set(x.replace("root:", "").split(":")) for x in [_gt, _pred]]
    recall = len(gt & pred) / len(gt)
    precision = len(gt & pred) / len(pred)
    return recall, precision


import networkx as nx


""" Bayesian network of GOLD
"""


def build_bayesian_onto(onto_net):
    "function to cronstruct a baysian network based on the probabilites derived from options"
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
    "get information accretion of o node in ontology"
    cond_prob = graph.nodes[term]["conditional_probability"]
    information_accretion = np.log2(1 / cond_prob)
    return information_accretion


def i_T(term, graph):
    """marginal probabilities that a protein is experimentally associated with a consistent subgraph T in the ontology.
    information content of a term (if str is provided) or a list of terms
    """
    if type(term) == str:
        veT = nx.ancestors(graph, term) | {term}
    else:
        veT = term
    _info_content = [ia_v(x, graph) for x in veT]
    return np.sum(_info_content)


def roc_preds(true, path, probs, threshold):  # noqa: D401 (simple helper)
    "Return path elements whose probability >= threshold (simple ROC helper)."
    return path[np.where(probs >= threshold)[0]]


def semantic_distance(ru, mi, k=2):
    "calculate diatnace"
    return ((ru**k) + (mi**k)) ** (1 / k)


def info_theoretic_metrics(T, P, graph):
    "get metrics based on the paper"
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
    # Check if both precision and recall are 0 to prevent division by zero.
    # This can occur when there are no positive predictions or positive labels
    if (beta**2 * precision) + recall == 0:
        return 0
    fscore = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    return fscore


def set_info_theoretic_metrics(true, pred, graph, k=1):
    "calculate info_theoretical metrics for a set of predictions"
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

