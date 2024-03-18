
# Copyright 2024 EMBL - European Bioinformatics Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01.00.01_baseData.ipynb.

# %% auto 0
__all__ = ['TAG', 'DATA_DIR', 'TMP_DIR', 'n_test', 'analysis_df_file', 'analysis_df', 'outputd_dicts', 'diamond_read',
           'krona_read', 'tax_annotations_from_file']

# %% ../nbs/01.00.01_baseData.ipynb 3
import glob
import json
import logging
import multiprocessing
import os
import re
import tarfile
import gzip
import pathlib
import mysql.connector
import pandas as pd
from tqdm import tqdm
from ssfMiscUtilities.generic import *

from . import config

# %% ../nbs/01.00.01_baseData.ipynb 4
from credentials import _credentials

# %% ../nbs/01.00.01_baseData.ipynb 5
TAG = 'baseData'

# %% ../nbs/01.00.01_baseData.ipynb 6
DATA_DIR = f"{config.datadir}/{TAG}"
TMP_DIR = f"{DATA_DIR}/temp"
os.makedirs(TMP_DIR,exist_ok=True)

# %% ../nbs/01.00.01_baseData.ipynb 8
n_test = False # number of lines to query for test porpuses

# %% ../nbs/01.00.01_baseData.ipynb 13
analysis_df_file = f"{DATA_DIR}/analysis_df.tsv.gz"

# %% ../nbs/01.00.01_baseData.ipynb 15
analysis_df = pd.read_csv(analysis_df_file,sep='\t')

# %% ../nbs/01.00.01_baseData.ipynb 27
def diamond_read(f):
    """ get taxonomy out of diamond functional annotation file
    """
    with gzip.open(f,'rb') as h:
        _diamonds = list({line.split(b'\t')[14].split(b'=')[-1].decode('utf8').replace('Candidatus ','') for line in h})
        edges = set()
        for s in _diamonds:
            spl = s.split()
            if bool(re.search("[A-Z]",spl[0][0])):
                edge = (
                    spl[0],s if len(spl)>1 else '' # seudo-graph in diamond, connect gr with sp
                )
                edges.add(edge)

        return list(edges)

def krona_read(content):
    """ function to read mseq.txt files
    """
    edges = set()  # Use a set to avoid duplicate edges
    for _line in content:
        line = _line.replace('Candidatus ','')
        # Split line and filter out any empty strings or strings representing empty nodes like 'k__'
        parts = [part for part in line.strip().split('\t') if part and not part.endswith('__')]
        # Skip the count at the beginning of each line
        lineage = parts[1:]
        
        # Initialize previous valid item variable
        prev = None
        for item in lineage:
            # Skip empty taxonomy levels
            if item.endswith('__'):
                continue
            if prev is not None:
                # Create an edge between the previous valid item and the current item
                prev1,prev2 = prev.split('__')
                item1,item2 = item.split('__')
                edges.add((prev1+'__'+prev2.split('__')[-1].replace('_',' ').replace('Candidatus ',''), item1+'__'+item2.split('__')[-1].replace('_',' ').replace('Candidatus ','') ))
            prev = item
    return list(edges)

def tax_annotations_from_file(f):
    """ fiunction to extract taxo_annots from file
    """
    d=None
    if 'diamond' in f:
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
                with gzip.open(f, 'rt') as content:  # 'rt' mode for reading text
                    d = krona_read(content)
            except Exception as e:
                print(f"An error occurred when loading diamond krona: {e}")
    return d
        

# %% ../nbs/01.00.01_baseData.ipynb 30
outputd_dicts = f'{TMP_DIR}/outdicts'
os.makedirs(outputd_dicts,exist_ok=True)
outputd_dicts
