from __future__ import annotations
from dataclasses import dataclass, field

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

    device: str | None = None
    max_length: int = 256
    threshold_rule: float | int | str = 0.01
    split_sentences: bool = False
    hf_model: str = "SantiagoSanchezF/trapiche-biome-classifier-text"
    model_version: str = "1.0"

@dataclass
class TaxonomyToVectorParams:
    """Configuration parameters for taxonomy vectorization.
    """
    hf_model: str = "SantiagoSanchezF/trapiche-biome-vectorizer-taxonomy"
    model_version: str = "1.0"

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
    In addition to model forward-pass settings, this also includes
    refinement and vectorizer data sources so downstream utilities can be
    invoked independently with a single params object.

    Fields
    ------
    batch_size
        Chunk size for model forward passes.
    k_knn
        Number of nearest neighbours used during refinement.
    dominance_threshold
        Frequency threshold for accepting lineage extensions in KNN refinement.
    vector_space
        Vector space identifier (currently only 'g' is supported).
    tru_column
        Column name in the metadata which contains the ground-truth lineage
        used by the refinement step (after formatting). Defaults to
        'BIOME_AMEND'.
    hf_model
        HF hub model id for the taxonomy-to-biome classifier.
    model_version
        Version tag for the taxonomy-to-biome classifier.
    """
    batch_size: int = 200
    k_knn: int = 20
    dominance_threshold: float = 0.5
    top_prob_diff_threshold: float = 0.05
    top_prob_ratio_threshold: float = 0.9
    hf_model: str = "SantiagoSanchezF/trapiche-biome-classifier-taxonomy"
    model_version: str = "1.0"

@dataclass
class TrapicheWorkflowParams:
    """Parameters for the Trapiche workflow.
    Attributes
    ----------
    run_text : bool
        Whether to run the text classification step.
    keep_text_results : bool
        Whether to keep text classification results in the output.
    run_vectorise : bool
        Whether to run the community vectorisation step.
    keep_vectorise_results : bool
        Whether to keep community vectorisation results in the output.
    run_taxonomy : bool
        Whether to run the taxonomy prediction step.
    keep_taxonomy_results : bool
        Whether to keep taxonomy prediction results in the output.
    output_keys : list[str] | None
        If a list, only these keys will be kept in the final output dicts. If None, keys are controled by keep_*_results config.
    
    """
    run_text: bool = True
    keep_text_results: bool = False
    run_vectorise: bool = True
    keep_vectorise_results: bool = False
    run_taxonomy: bool = True
    keep_taxonomy_results: bool = False
    output_keys: list[str] | None = field(default_factory=lambda: ["text_predictions", "raw_unambiguous_prediction", "raw_refined_prediction","constrained_unambiguous_prediction", "constrained_refined_prediction", "final_selected_prediction"]) 
    