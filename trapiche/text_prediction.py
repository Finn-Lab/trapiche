"""Text classification utilities (multi-label) for biome prediction from descriptions.

This module exposes pure functions for multi‑label BERT classification plus a CLI
entry point. It implements:

- Automatic device selection (CUDA if available, else CPU).
- Lazy, cached model loading using Hugging Face Transformers.
- Optional sentence splitting using spaCy (or NLTK / naive fallback).
- Optional class specific probability thresholds loaded from ``class_thresholds.json``
    when present alongside the model weights (local directory or resolved cache path).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Iterable, List, Mapping, Sequence, Tuple
from functools import lru_cache

import numpy as np
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
from safetensors.torch import load_file


from .utils import _get_hf_model_path
import logging
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Device selection and model loading
# ---------------------------------------------------------------------
def choose_device(device: str | None = None) -> str:
    """Return the best available device string.

    If a specific device is provided, it's returned unchanged. Otherwise, this
    function selects CUDA when available, else CPU.
    """
    if device:
        return str(device)
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # pragma: no cover - conservative
        pass
    return "cpu"


@lru_cache(maxsize=2)
def load_text_model(model_name: str, model_version:str, device: str | None = None) -> Tuple[
    BertTokenizerFast, BertConfig, Mapping[int, str], BertForSequenceClassification, Mapping[str, float]
]:
    """Lazily load and cache tokenizer, config, model, and thresholds.

    Returns (tokenizer, config, id2label, model, thresholds).
    """
    dev = choose_device(device)

    vocab_path = _get_hf_model_path(model_name, model_version, "vocab_*.txt")
    tokenizer_json_path = _get_hf_model_path(model_name, model_version, "tokenizer_*.json")
    tokenizer_config_path = _get_hf_model_path(model_name, model_version, "tokenizer_config_*.json")
    special_tokens_map_path = _get_hf_model_path(model_name, model_version, "special_tokens_map_*.json")
    config_path = _get_hf_model_path(model_name, model_version, "config_*.json")
    model_weights_path = _get_hf_model_path(model_name, model_version, "model_*.safetensors")

    with open(tokenizer_config_path) as h:
        tokenizer_config = json.load(h)

    tokenizer = BertTokenizerFast(
        vocab_file=vocab_path,
    )

    config = BertConfig.from_json_file(config_path)

    id2label = {int(k):v for k,v in config.id2label.items()}

    model = BertForSequenceClassification(config)

    state_dict = load_file(model_weights_path)
    model.load_state_dict(state_dict, strict=False)

    try:  # pragma: no cover - trivial
        model.to(dev)  
    except Exception:
        pass
    model.eval()

    thresholds = _load_thresholds_multi_path(model_name, tokenizer, model)
    return tokenizer, config, id2label, model, thresholds


def _load_thresholds_multi_path(
    model_path: str,
    tokenizer: BertTokenizerFast | None = None,
    model: BertForSequenceClassification | None = None,
) -> Mapping[str, float]:
    """Attempt to locate class_thresholds.json near the resolved model files."""
    candidates: List[str] = []
    if os.path.isdir(model_path):
        candidates.append(model_path)
    for obj in (tokenizer, getattr(model, "config", None), model):
        if obj is None:
            continue
        p = getattr(obj, "name_or_path", None) or getattr(obj, "_name_or_path", None)
        if isinstance(p, str) and os.path.isdir(p):
            candidates.append(p)

    for base in candidates:
        path = os.path.join(base, "class_thresholds.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        # Ensure numeric floats
                        return {str(k): float(v) for k, v in data.items()}
            except Exception:
                continue
    return {}


# ---------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------
_NLP = None  # module-level cache for spaCy model


def _get_nlp():  # pragma: no cover - optional path
    global _NLP
    if _NLP is not None:
        return _NLP
    try:  # optional dependency
        import spacy  # type: ignore
    except Exception:
        _NLP = None
        return None
    # Try SciSpaCy first then general English
    for model_name in ("en_core_sci_sm", "en_core_web_sm"):
        try:
            _NLP = spacy.load(model_name)  # type: ignore
            break
        except Exception:
            try:
                spacy.cli.download(model_name)  # type: ignore
                _NLP = spacy.load(model_name)  # type: ignore
                break
            except Exception:
                continue
    return _NLP


def _split_sentences(text: str) -> List[str]:
    nlp = _get_nlp()
    if nlp is not None:
        try:
            return [s.text.strip() for s in nlp(text).sents if s.text.strip()]
        except Exception:
            pass
    # Fallbacks
    try:  # nltk fallback
        import nltk  # type: ignore
        from nltk.tokenize import sent_tokenize  # type: ignore

        nltk.download("punkt", quiet=True)  # idempotent
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except Exception:
        return [s.strip() for s in text.split('.') if s.strip()]


# ---------------------------------------------------------------------
# Public API (pure functions)
# ---------------------------------------------------------------------
def predict_probability(
    texts: Sequence[str] | str,
    model_name: str,
    model_version: str,
    device: str | None = None,
    max_length: int = 256,
) -> np.ndarray:
    if isinstance(texts, str):  # type: ignore
        texts = [texts]  # type: ignore
    tokenizer, _config, _id2label, model, _thresholds = load_text_model(model_name, model_version, device)
    enc = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    dev = choose_device(device)
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        probs = torch.sigmoid(out.logits).cpu().numpy()
    return probs


def probabilities_to_mask(
    probs: np.ndarray,
    id2label: Mapping[int, str],
    thresholds: Mapping[str, float],
    rule: str | float | int = 0.01,
) -> np.ndarray:
    n_samples, n_classes = probs.shape
    mask = np.zeros_like(probs, dtype=int)

    # Build per-class thresholds (if numeric rule)
    per_class = None
    if isinstance(rule, (int, float)):
        per_class = [thresholds.get(id2label.get(i, f"label_{i}"), float(rule)) for i in range(n_classes)]

    for i, row in enumerate(probs):
        if rule == "max":
            mask[i, int(np.argmax(row))] = 1
        elif isinstance(rule, str) and rule.startswith("top-"):
            try:
                top_n = int(rule.split('-')[1])
            except ValueError as e:  # pragma: no cover - defensive
                raise ValueError("Invalid top-N rule format: use 'top-N'.") from e
            top_idx = np.argsort(row)[::-1][:top_n]
            mask[i, top_idx] = 1
        elif per_class is not None:
            for j, p in enumerate(row):
                if p >= per_class[j]:
                    mask[i, j] = 1
        else:  # pragma: no cover - unexpected
            raise ValueError("Unsupported threshold rule.")
    return mask


def predict(
    texts: Sequence[str] | str,
    model_name: str,
    model_version: str,
    device: str | None = None,
    max_length: int = 256,
    threshold_rule: str | float | int = 0.01,
    split_sentences: bool = False,
) -> List[List[str]]:
    if isinstance(texts, str):
        texts_list = [texts]
    else:
        texts_list = list(texts)
    tokenizer, _config, id2label, _model, thresholds = load_text_model(model_name, model_version, device)
    # Unused tokenizer variable purposefully retained to ensure cache priming above
    _ = tokenizer  # pragma: no cover

    if split_sentences:
        agg = []
        for t in texts_list:
            parts = _split_sentences(t)
            parts.append(t)  # include full context
            p = predict_probability(parts, model_name=model_name, model_version=model_version, device=device, max_length=max_length)
            agg.append(p.max(axis=0))
        probs = np.vstack(agg) if agg else np.zeros((0, len(id2label)))
    else:
        probs = predict_probability(texts_list, model_name=model_name, model_version=model_version, device=device, max_length=max_length)

    mask = probabilities_to_mask(probs, id2label=id2label, thresholds=thresholds, rule=threshold_rule)
    predictions: List[List[str]] = []
    for row in mask:
        labels = [id2label.get(i, f"label_{i}") for i, v in enumerate(row) if v == 1]
        predictions.append(labels)
    return predictions
