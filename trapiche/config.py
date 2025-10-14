from __future__ import annotations
from dataclasses import dataclass

@dataclass
class TaxonomyToBiomeParams:
    """Configuration parameters for deep lineage prediction.

    Attributes
    ----------
    batch_size : int
        Chunk size for model forward passes.
    k_knn : int
        Number of nearest neighbours used during refinement.
    dominance_threshold : float
        Frequency threshold for accepting lineage extensions in KNN refinement.
    """
    batch_size: int = 200
    k_knn: int = 10
    dominance_threshold: float = 0.5


@dataclass
class TextToBiomeParams:
    """Configuration parameters for text biome prediction.

    Attributes
    ----------
    model_path : str
        HF hub model id or local directory containing a fine‑tuned model.
    device : str | None
        Explicit device (e.g. 'cpu', 'cuda', 'cuda:0'). If None, auto-select.
    max_length : int
        Maximum token length for encoding input text(s).
    threshold_rule : float | int | str
        Thresholding rule: numeric, 'max', or 'top-N' (e.g. 'top-3').
    split_sentences : bool
        When True, split each input text into sentences and aggregate with max.
    """

    model_path: str = "SantiagoSanchezF/trapiche-biome-classifier"
    device: str | None = None
    max_length: int = 256
    threshold_rule: float | int | str = 0.01
    split_sentences: bool = False

@dataclass
class TrapicheWorkflowParams:

    run_text: bool = True
    keep_text_results: bool = False
    run_vectorise: bool = True
    keep_vectorise_results: bool = False
    run_taxonomy: bool = True
    keep_taxonomy_results: bool = False
    output_keys: list[str] | None = ["text_predictions", "lineage_prediction","lineage_prediction_probability","refined_prediction"]
    