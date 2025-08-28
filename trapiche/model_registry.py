"""Model registry, cache management, and download utilities for Trapiche.

Minimal, pragmatic implementation:
 - Central MODEL_REGISTRY mapping (name, version) -> metadata (url/sha256/size/provider)
 - Cache root resolved from TRAPICHE_CACHE env var or ~/.cache/trapiche
 - Public API:
     get_model_path(name, version='1.0', auto_download=False) -> local path (may raise)
     ensure_model(name, version='1.0') -> local path (downloading if missing)
     download_all(progress=True)
 - Resumable, streaming downloads with progress bar (tqdm) and atomic rename
 - SHA256 integrity verification (optional if hash provided)
 - HuggingFace models: rely on transformers caching; we just trigger a warm download
 - SciSpacy model: install via pip if not present
 - CLI integration added in cli.py (trapiche download-models)

This keeps scope intentionally small while following good practices.
"""

from __future__ import annotations

import hashlib
import os
import sys
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import requests
from tqdm import tqdm

DEFAULT_VERSION = "1.0"

logger = logging.getLogger(__name__)

@dataclass
class ModelMeta:
    name: str
    version: str
    url: Optional[str] = None  # For direct (FTP/HTTP) downloads
    sha256: Optional[str] = None
    size: Optional[int] = None
    provider: str = "ftp"  # ftp | huggingface | pip
    note: str = ""


# Central registry. Extend here to add new models.
MODEL_REGISTRY: Dict[Tuple[str, str], ModelMeta] = {
    ("full_final_taxonomy.model.keras", "1.0"): ModelMeta(
        name="full_final_taxonomy.model.keras",
        version="1.0",
        url="https://ftp.ebi.ac.uk/pub/databases/metagenomics/trapiche/models/full_final_taxonomy.model.keras",
        sha256="21b032020e30de6f60147cf51ff1e100f8544f25c8ec4d808f67acab172ad298",
        size=30030544344,
        provider="ftp",
        note="Keras taxonomy deep model",
    ),
    ("mgnify_sample_vectors_v1.0.h5", "1.0"): ModelMeta(
        name="mgnify_sample_vectors",
        version="1.0",
        url="https://ftp.ebi.ac.uk/pub/databases/metagenomics/trapiche/models/mgnify_sample_vectors.h5",
        sha256="4bdd9bafdaa019b101d30322dbf2fd5f3dc787f284aac9ea942f645a92705aa3",
        size=212697584,
        provider="ftp",
        note="Community embedding matrix",
    ),
    # Logical / virtual entries for external providers
    ("trapiche-biome-classifier", "1.0"): ModelMeta(
        name="trapiche-biome-classifier",
        version="1.0",
        provider="huggingface",
        note="HuggingFace model: SantiagoSanchezF/trapiche-biome-classifier",
    ),
    ("en_core_sci_sm", "0.5.4"): ModelMeta(
        name="en_core_sci_sm",
        version="0.5.4",
        provider="pip",
        url="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz",
        note="SciSpacy small English biomedical model",
    ),
}


def cache_root() -> Path:
    logger.debug("cache_root called")
    root = os.environ.get("TRAPICHE_CACHE")
    if not root:
        root = os.path.join(Path.home(), ".cache", "trapiche")
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    logger.info(f"cache_root path={p}")
    return p


def model_cache_dir(name: str, version: str = DEFAULT_VERSION) -> Path:
    d = cache_root() / name / version
    d.mkdir(parents=True, exist_ok=True)
    return d


def _hash_file(path: Path, algo: str = "sha256", chunk: int = 1 << 20) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _download_stream(url: str, dest: Path, expected_size: Optional[int], expected_sha256: Optional[str]):
    # Support resume
    headers = {}
    temp_path = dest.with_suffix(dest.suffix + ".part")
    pos = 0
    if temp_path.exists():
        pos = temp_path.stat().st_size
        headers["Range"] = f"bytes={pos}-"

    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        total = None
        if expected_size is not None:
            total = expected_size
        elif 'Content-Length' in r.headers:
            total = int(r.headers['Content-Length']) + pos
        mode = "ab" if pos else "wb"
        with temp_path.open(mode) as f, tqdm(
            total=total, initial=pos, unit="B", unit_scale=True, desc=f"Downloading {dest.name}", disable=not sys.stderr.isatty()
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    # Integrity check
    if expected_sha256:
        sha = _hash_file(temp_path)
        if sha != expected_sha256:
            raise ValueError(f"SHA256 mismatch for {dest.name}: {sha} != {expected_sha256}")

    temp_path.replace(dest)  # atomic on POSIX


def _ensure_ftp_model(meta: ModelMeta) -> Path:
    target = model_cache_dir(meta.name, meta.version) / meta.name
    if target.exists():
        # Optionally verify hash if provided
        if meta.sha256 and _hash_file(target) != meta.sha256:
            target.unlink()
        else:
            return target
    if not meta.url:
        raise ValueError(f"No URL for model {meta.name}")
    _download_stream(meta.url, target, meta.size, meta.sha256)
    return target


def _ensure_huggingface_model(meta: ModelMeta) -> Path:
    """Trigger download via transformers; rely on its cache. Returns cache dir."""
    try:
        from transformers import BertTokenizerFast
    except ImportError as e:
        raise RuntimeError("transformers required for HuggingFace models: pip install transformers") from e

    # Model id is assumed to be SantiagoSanchezF/trapiche-biome-classifier
    model_id = "SantiagoSanchezF/trapiche-biome-classifier"
    # We only need to trigger one file load to populate the cache
    BertTokenizerFast.from_pretrained(model_id)
    # Find huggingface cache location
    hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return hf_cache


def _ensure_pip_model(meta: ModelMeta) -> Path:
    import importlib
    try:
        importlib.import_module(meta.name)
        return Path(importlib.util.find_spec(meta.name).origin).parent
    except Exception:
        # install from URL
        if not meta.url:
            raise RuntimeError(f"No URL to install {meta.name}")
        import subprocess
        cmd = [sys.executable, "-m", "pip", "install", meta.url]
        subprocess.check_call(cmd)
        import importlib
        m = importlib.import_module(meta.name)
        return Path(m.__file__).parent


def ensure_model(name: str, version: str = DEFAULT_VERSION) -> Path:
    meta = MODEL_REGISTRY.get((name, version))
    if not meta:
        raise KeyError(f"Model {name} version {version} not in registry.")
    if meta.provider == "ftp":
        return _ensure_ftp_model(meta)
    if meta.provider == "huggingface":
        return _ensure_huggingface_model(meta)
    if meta.provider == "pip":
        return _ensure_pip_model(meta)
    raise ValueError(f"Unknown provider {meta.provider}")


def get_model_path(name: str, version: str = DEFAULT_VERSION, auto_download: bool = False) -> Path:
    """Return path to model file or directory.
    If missing and auto_download is False, raises FileNotFoundError with guidance.
    """
    meta = MODEL_REGISTRY.get((name, version))
    if not meta:
        raise KeyError(f"Model {name} version {version} not found in registry.")
    cache_dir = model_cache_dir(name, version)
    if meta.provider == "ftp":
        candidate = cache_dir / meta.name
        if candidate.exists():
            return candidate
        if auto_download:
            return ensure_model(name, version)
        raise FileNotFoundError(
            f"Model file {candidate} missing. Use trapiche.download_models() or CLI 'trapiche-download-models'."
        )
    if meta.provider in {"huggingface", "pip"}:
        # We don't replicate HF/pip assets inside our cache; we just check availability
        try:
            return ensure_model(name, version) if auto_download else ensure_model(name, version)
        except Exception as e:
            if auto_download:
                raise
            raise FileNotFoundError(f"External model {name} not available: {e}") from e
    raise ValueError(f"Unhandled provider {meta.provider}")


def download_all(progress: bool = True):
    """Download all registered models (ftp & trigger others)."""
    errors = {}
    for (name, version), meta in MODEL_REGISTRY.items():
        try:
            ensure_model(name, version)
        except Exception as e:
            errors[(name, version)] = str(e)
    if errors:
        raise RuntimeError(f"Some models failed: {errors}")


# Convenience aliases for external consumption
def download_models():  # simple, stable public API
    logger.info("download_models called")
    download_all()
    logger.info("download_models finished")
