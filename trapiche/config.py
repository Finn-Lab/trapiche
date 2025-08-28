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
