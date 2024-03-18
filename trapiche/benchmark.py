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
# See the License for the specific la
# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/BENCHMARK.ipynb.

# %% auto 0
__all__ = []

# %% ../nbs/BENCHMARK.ipynb 3
import glob
import json
import os
from . import baseData
from tqdm import tqdm
from collections import Counter
import numpy as np
import pandas as pd
import shutil
import re
from ssfMiscUtilities.generic import *

from . import config

# %% ../nbs/BENCHMARK.ipynb 14
from credentials import _credentials
import mysql.connector
