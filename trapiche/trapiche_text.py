"""Text classification (multi-label) wrapper for biome prediction from study descriptions.

This module provides a thin convenience layer around a multi‑label BERT classifier
plus a CLI entry point. Optional sentence splitting uses spaCy (or NLTK / naive
fallback). Class specific probability thresholds are loaded from an optional
JSON file named ``class_thresholds.json`` inside the model directory (or HF hub
cache path).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Iterable, List, Sequence

import numpy as np
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

import logging
logger = logging.getLogger(__name__)


__all__ = ["TextClassifier", "main"]


class TextClassifier:
    """Multi‑label text classifier wrapper.

    Parameters
    ----------
    model_path: str
        HF hub model id or local directory containing a fine‑tuned model.
    device: str | None
        Explicit device ("cpu" / "cuda" / "cuda:0"). If ``None`` an available
        CUDA device is used, else CPU.
    """

    def __init__(self, model_path: str, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.model_config = BertConfig.from_pretrained(model_path)
        num_labels = getattr(self.model_config, "num_labels", 0)
        self.id2label = getattr(
            self.model_config,
            "id2label",
            {i: f"label_{i}" for i in range(num_labels)},
        )
        self.model = BertForSequenceClassification.from_pretrained(
            model_path, config=self.model_config
        )
        # Move model to device via small wrapper to appease some static analyzers
        self._move_to_device()
        self.model.eval()
        self.thresholds = self._load_thresholds(model_path)
        self._nlp = None  # spaCy model cache

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _load_thresholds(self, model_path: str) -> dict:
        path = os.path.join(model_path, "class_thresholds.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _get_nlp(self):  # pragma: no cover - optional path
        if self._nlp is not None:
            return self._nlp
        try:  # optional dependency
            import spacy  # type: ignore
        except Exception:
            self._nlp = None
            return None
        # Try SciSpaCy first then general English
        for model_name in ("en_core_sci_sm", "en_core_web_sm"):
            try:
                self._nlp = spacy.load(model_name)  # type: ignore
                break
            except Exception:
                try:
                    spacy.cli.download(model_name)  # type: ignore
                    self._nlp = spacy.load(model_name)  # type: ignore
                    break
                except Exception:
                    continue
        return self._nlp

    def _split_sentences(self, text: str) -> List[str]:
        nlp = self._get_nlp()
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

    # Device helper -----------------------------------------------------
    def _move_to_device(self) -> None:
        """Safely move model to target device.

        Wrapped to avoid over-eager static analyzers misinterpreting the
        ``.to`` invocation as a call on the *class* rather than the instance.
        Silently ignores failures (e.g., if a meta device is used in tests).
        """
        try:  # pragma: no cover - trivial
            self.model.to(self.device)  # type: ignore[call-arg]
        except Exception:
            pass

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def predict_probability(
        self, texts: Sequence[str], max_length: int = 256
    ) -> np.ndarray:
        if isinstance(texts, str):  # type: ignore
            texts = [texts]  # type: ignore
        enc = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
            probs = torch.sigmoid(out.logits).cpu().numpy()
        return probs

    def _probabilities_to_mask(
        self, probs: np.ndarray, rule: str | float | int = 0.01
    ) -> np.ndarray:
        n_samples, n_classes = probs.shape
        mask = np.zeros_like(probs, dtype=int)

        # Build per-class thresholds (if numeric rule)
        per_class = None
        if isinstance(rule, (int, float)):
            per_class = []
            for i in range(n_classes):
                label = self.id2label.get(i, f"label_{i}")
                per_class.append(self.thresholds.get(label, float(rule)))

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
        self,
        texts: Sequence[str] | str,
        max_length: int = 256,
        threshold_rule: str | float | int = 0.01,
        split_sentences: bool = False,
    ) -> List[List[str]]:
        if isinstance(texts, str):
            texts_list = [texts]
        else:
            texts_list = list(texts)

        if split_sentences:
            agg = []
            for t in texts_list:
                parts = self._split_sentences(t)
                parts.append(t)  # include full context
                p = self.predict_probability(parts, max_length=max_length)
                agg.append(p.max(axis=0))
            probs = np.vstack(agg)
        else:
            probs = self.predict_probability(texts_list, max_length=max_length)

        mask = self._probabilities_to_mask(probs, threshold_rule)
        predictions: List[List[str]] = []
        for row in mask:
            labels = [self.id2label.get(i, f"label_{i}") for i, v in enumerate(row) if v == 1]
            predictions.append(labels)
        return predictions


def main(argv: Iterable[str] | None = None) -> None:  # CLI helper
    parser = argparse.ArgumentParser(
        description="Multi-label biome text classifier"
    )
    parser.add_argument(
        "--model_path",
        default="SantiagoSanchezF/trapiche-biome-classifier",
        help="HF hub model id or local directory",
    )
    parser.add_argument(
        "--input_text",
        required=True,
        help="Raw text or path to JSON list file",
    )
    parser.add_argument(
        "--output_file",
        help="Optional path to write predictions as JSON",
    )
    parser.add_argument(
        "--threshold",
        default=0.01,
        help="Threshold rule: float/int, 'max', or 'top-N' (e.g. top-3)",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Enable sentence splitting aggregation",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    clf = TextClassifier(args.model_path)

    if os.path.isfile(args.input_text) and args.input_text.endswith(".json"):
        with open(args.input_text, "r", encoding="utf-8") as fh:
            texts = json.load(fh)
    else:
        texts = [args.input_text]

    preds = clf.predict(
        texts,
        threshold_rule=args.threshold,  # type: ignore[arg-type]
        split_sentences=args.split,
    )

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as fh:
            json.dump(preds, fh, indent=2)
    else:
        print(json.dumps(preds, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
