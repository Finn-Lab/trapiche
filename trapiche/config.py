import logging
logger = logging.getLogger(__name__)
def load_config():
	logger.info("load_config called")
	# ...existing code...
	logger.info("config loaded")
import os
import pathlib

import trapiche as _trapiche_pkg

basedir = pathlib.Path(os.path.dirname(_trapiche_pkg.__file__))
datadir = basedir / "data"
datadir.mkdir(parents=True, exist_ok=True)
