"""Taxonomy tree loading and traversal helpers."""

# %% auto 0
__all__ = ['TAG', 'DATA_DIR', 'TMP_DIR', 'taxonomy_graph_file', 'taxonomy_graph', 'taxonomy_composition_dir', 'get_subgraph',
           'get_terminal_nodes']

# %% ../nbs/01.00.02_taxonomyTree.ipynb 3
import os

from . import config


# %% ../nbs/01.00.02_taxonomyTree.ipynb 4
TAG = "taxonomyTree"


# %% ../nbs/01.00.02_taxonomyTree.ipynb 5
DATA_DIR = f"{config.datadir}/{TAG}"
TMP_DIR = f"{DATA_DIR}/temp"
os.makedirs(TMP_DIR, exist_ok=True)


# %% ../nbs/01.00.02_taxonomyTree.ipynb 10
import pandas as pd
import networkx as nx


# %% ../nbs/01.00.02_taxonomyTree.ipynb 26
taxonomy_graph_file = f"{DATA_DIR}/taxonomy_graph.graphml"


# %% ../nbs/01.00.02_taxonomyTree.ipynb 29
taxonomy_graph = nx.read_graphml(taxonomy_graph_file, node_type=str)


# %% ../nbs/01.00.02_taxonomyTree.ipynb 30
def get_subgraph(nodes):
    "given a set of nodes in the taxonomy_graph get the subgraph"
    nmix = set()
    for x in nodes:
        # _x = _x.replace('__','//').replace('_',' ').replace('//','__').replace('Candidatus ','')
        # x = ter_gtdb_backbone_ids.get( other2gtdb.get(_x,_x) )
        # if x==None:
        # continue

        _cu = [[x, 0]]
        while len(_cu) != 0:
            cu = _cu[0][0]
            if cu in nmix:
                break
            nmix |= {cu}
            _cu = list(taxonomy_graph.in_edges(cu))
    H = taxonomy_graph.subgraph(nmix)
    return H


# %% ../nbs/01.00.02_taxonomyTree.ipynb 31
def get_terminal_nodes(nodes):
    "given a set of nodes in the taxonomy_graph get the terminal nodes AKA leaves"
    H = get_subgraph(nodes)
    leaves = [x for x in H.nodes() if H.out_degree(x) == 0 and H.in_degree(x) == 1]
    return leaves


# %% ../nbs/01.00.02_taxonomyTree.ipynb 39
taxonomy_composition_dir = f"{TMP_DIR}/taxonomy_composition"

