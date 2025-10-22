from ._version import version as __version__

import logging

# Provide a NullHandler so that if an application using this package does
# not configure logging, library messages are silently ignored by default
# (best practice for libraries).
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Provide lightweight lazy wrappers so callers can configure logging via
# `trapiche.setup_logging(...)` or obtain a package logger with
# `trapiche.get_logger()`. The implementation imports the helper module
# only when the functions are called which avoids static import resolution
# issues in constrained analysis environments.

def setup_logging(level: int = logging.INFO, fmt: None | str = None) -> None:
	"""Configure package logging using the bundled helper.

	This function lazily imports `trapiche.logging_config` so importing the
	package never fails in environments where imports can't be resolved by
	static analyzers.
	"""
	try:
		from .logging_config import setup_logging as _setup  # type: ignore

		return _setup(level=level, fmt=fmt)
	except Exception:
		# If for any reason the helper isn't available, fail silently; the
		# package is safe to import and will use the NullHandler added above.
		return None


def get_logger(name: str | None = None) -> logging.Logger:
	"""Return the package logger (or a named child)."""
	try:
		from .logging_config import get_logger as _get  # type: ignore

		return _get(name)
	except Exception:
		return logging.getLogger(__name__)

__all__ = ["__version__", "setup_logging", "get_logger"]
