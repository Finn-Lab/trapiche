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

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00.0.0_config.ipynb.

# %% auto 0
__all__ = ['module_name', 'is_notebook', 'datadir']

# %% ../nbs/00.0.0_config.ipynb 2
from os import path,makedirs

# %% ../nbs/00.0.0_config.ipynb 3
module_name='trapiche'

# %% ../nbs/00.0.0_config.ipynb 4
is_notebook = False
try:
    is_notebook = get_ipython().__class__.__name__
except:
    None

# %% ../nbs/00.0.0_config.ipynb 5
if is_notebook == False:
    basedir =  path.abspath(path.join(path.dirname(__file__),module_name))
else:
    basedir =  path.abspath(path.join("..",module_name))

# %% ../nbs/00.0.0_config.ipynb 6
datadir = path.join(basedir,'data')
makedirs(datadir,exist_ok=True)
