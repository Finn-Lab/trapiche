"""Base dataset utilities and taxonomy file parsing helpers for Trapiche.

Previously generated via notebooks; cleaned for distribution. Provides
helpers to extract taxonomy annotations from diamond and krona style files.
"""

__all__ = ['TAG', 'DATA_DIR', 'TMP_DIR', 'n_test', 'analysis_df_file', 'analysis_df', 'diamond_read',
           'krona_read', 'tax_annotations_from_file']

import os
import re
import gzip
from pathlib import Path
from contextlib import contextmanager
import pandas as pd

from . import config


TAG = "baseData"


DATA_DIR = f"{config.datadir}/{TAG}"
TMP_DIR = f"{DATA_DIR}/temp"
os.makedirs(TMP_DIR, exist_ok=True)


n_test = False  # number of lines to query for test purposes (debugging flag)


analysis_df_file = f"{DATA_DIR}/analysis_df.tsv.gz"


analysis_df = pd.read_csv(analysis_df_file, sep="\t")


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
    """fiunction to extract taxo_annots from file"""
    d = None
    if "diamond" in f:
        try:
            d = diamond_read(f)
        except Exception as e:
            print(f"An error occurred when loading diamond file: {e}")
    else:
        try:
            with open(f) as content:
                d = krona_read(content)
        except:
            try:
                with gzip.open(f, "rt") as content:  # 'rt' mode for reading text
                    d = krona_read(content)
            except Exception as e:
                print(f"An error occurred when loading diamond krona: {e}")
    return d
