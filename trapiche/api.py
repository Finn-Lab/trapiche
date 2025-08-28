from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Any
import numpy as np

from .config import TaxonomyToBiomeParams
from .community2vec import vectorise_sample

class Community2vec:
    def __init__(self):
        pass
    def transform(self, list_of_tax_files) -> np.ndarray:
        """Vectorise one or many samples from taxonomy annotation files.

        Flexible input forms are accepted (all are normalised internally to the
        original list-of-lists of file paths interface):

        1. Single file path (str / Path): that file represents one sample.
        2. Directory path: every *.tsv or *.tsv.gz file inside (non-recursive)
           is treated as belonging to one sample.
        3. Flat list of file paths: all those files together are one sample.
        4. List of lists of file paths (original behaviour): each inner list is a sample.

        Returns
        -------
        np.ndarray
            Array with shape (n_samples, embedding_dim). If no vectors can be
            produced an array with shape (n_samples, 0) is returned (or (0, 0) if
            there are no samples at all).
        """
        self.vectorised_samples = vectorise_sample(list_of_tax_files)
        return self.vectorised_samples
    
    def save(self, path: str | Path) -> None:
        """Save the vectorised samples to a .npy file."""
        if not hasattr(self, 'vectorised_samples'):
            raise ValueError("No vectorised samples to save. Call transform() first.")
        np.save(path, self.vectorised_samples)


class TaxonomyToBiome:
    """Lightweight wrapper around deep taxonomy prediction.

    Mirrors the minimal style of other API classes. All heavy imports are
    deferred until prediction time to avoid increasing import cost.
    """

    def __init__(self, params: TaxonomyToBiomeParams | None = None):
        self.params = params or TaxonomyToBiomeParams()

    def predict(
        self,
        list_of_tax_files: Sequence[Any],
        constrain: Sequence[Sequence[str]] | None = None,
        return_full_preds: bool = False,
        params: TaxonomyToBiomeParams | None = None,
    ):
        """Run deep biome lineage prediction.

        Parameters
        ----------
        list_of_tax_files : Sequence
            Accepts the same flexible inputs as Community2vec.transform (a
            list-of-lists is passed directly; other forms are normalised
            internally by Community2vec inside the underlying implementation).
        constrain : sequence of sequence of str, optional
            Per-sample potential space constraints; pass None for unconstrained.
        return_full_preds : bool, default False
            When True, returns the full probability dataframe, else a reduced view.
        params : DeepPredictorParams, optional
            Override instance parameters for this call only.

        Returns
        -------
        (pd.DataFrame, np.ndarray)
            Predictions dataframe and the sample embedding matrix.
        """
        _params = params or self.params
        # Local import to avoid circular dependency (deep_pred imports Community2vec from this module)
        from . import deep_pred  # type: ignore

        self.df, self.vec = deep_pred.predict_runs(
            list_of_list=list_of_tax_files,
            return_full_preds=return_full_preds,
            constrain=constrain,
            batch_size=_params.batch_size,
            k_knn=_params.k_knn,
            dominance_threshold=_params.dominance_threshold,
        )
        return self.df, self.vec
    
    def save_vectors(self, path: str | Path) -> None:
        """Save the vectorised samples to a .npy file."""
        np.save(path, self.vec)

    def save(self, path: str | Path) -> None:
        """Save the predictions dataframe to a CSV file."""
        self.df.to_csv(path, index=False)


