import importlib.resources as resources
import pathlib
import os

import trapiche

basedir = pathlib.Path(resources.files(trapiche))
datadir = path.join(basedir, "data")
os.makedirs(datadir, exist_ok=True)
