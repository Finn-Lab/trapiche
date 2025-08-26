import os
import pathlib

import trapiche as _trapiche_pkg

basedir = pathlib.Path(os.path.dirname(_trapiche_pkg.__file__))
datadir = basedir / "data"
datadir.mkdir(parents=True, exist_ok=True)
