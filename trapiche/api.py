from pathlib import Path
from typing import List, Sequence, Any
import json
import numpy as np

from .config import TaxonomyToBiomeParams, TextToBiomeParams
from .config import TrapicheWorkflowParams
from .workflow import run_workflow
from typing import Dict, Any, Sequence

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
        from .taxonomy_vectorization import vectorise_sample

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
        community_vectors: np.ndarray,
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
        # Local import to avoid circular dependency (taxonomy_prediction imports Community2vec from this module)
        from . import taxonomy_prediction  # type: ignore

        self.df, self.vec = taxonomy_prediction.predict_runs(
            community_vectors=community_vectors,
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


class TextToBiome:
    """Lightweight wrapper around the text classifier.

    Mirrors the minimal style of other API classes. Parameters are grouped in
    a dataclass, and heavy imports are deferred until prediction time.
    """

    def __init__(self, params: TextToBiomeParams | None = None) -> None:
        self.params = params or TextToBiomeParams()
        self.predictions_: List[List[str]] | None = None  # last predictions (optional convenience)

    def predict(
        self,
        texts: Sequence[str] | str,
        params: TextToBiomeParams | None = None,
    ) -> List[List[str]]:
        """Run text-based biome prediction.

        Parameters
        ----------
        texts : Sequence[str] | str
            One or more input texts.
        params : TextToBiomeParams, optional
            Override instance parameters for this call only.
        """
        # Local import to defer transformers and friends until needed
        from . import text_prediction as tt  # type: ignore

        _p = params or self.params
        preds = tt.predict(
            texts,
            model_path=_p.model_path,
            device=_p.device,
            max_length=_p.max_length,
            threshold_rule=_p.threshold_rule,
            split_sentences=_p.split_sentences,
        )
        self.predictions_ = preds
        return preds

    def save(self, path: str | Path) -> None:
        """Save the latest predictions to a JSON file.

        Note: This is a convenience method mirroring other wrappers; call
        predict() first to populate predictions_.
        """
        if self.predictions_ is None:
            raise ValueError("No predictions to save. Call predict() first.")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.predictions_, f, indent=2)


class TrapicheWorkflowFromSequence:
    """Lightweight wrapper that runs the in-repo `run_workflow` on a sequence
    of sample dicts and returns the augmented sequence.

    The class mirrors the simple API style used elsewhere in this module and
    accepts an optional `TrapicheWorkflowParams` instance to control which
    steps run and whether intermediate results are kept.
    """

    def __init__(self, params: TrapicheWorkflowParams | None = None) -> None:
        self.params = params or TrapicheWorkflowParams()

    def run(self, samples: Sequence[Dict[str, Any]]) -> Sequence[Dict[str, Any]]:
        """Run the workflow on `samples` and return the augmented list.

        Parameters
        ----------
        samples : Sequence[Dict[str, Any]]
            Each dict should contain optional keys `project_description_file_path`
            and `taxonomy_files_paths` (list). The function returns a new list
            of dicts (shallow copies) augmented with results per sample.
        """

        result = run_workflow(samples, run_text=self.params.run_text, run_vectorise=self.params.run_vectorise, run_taxonomy=self.params.run_taxonomy)

        # process which keys to keep according to config
        keep_keys = set()
        sample_keys = set(samples[0].keys()) if samples else set()
        all_keys = set(result[0].keys()) if result else set()
        if self.params.output_keys:
            keep_keys = set(self.params.output_keys)
        else:
            keep_keys = set(result[0].keys()) if result else set()
            if not self.params.keep_text_results:
                keep_keys.discard("text_predictions")
            if not self.params.keep_vectorise_results:
                keep_keys.discard("community_vector")
            if not self.params.keep_taxonomy_results:
                # TODO: add a tag to the taxonomy results dict to identify its keys
                taxonomy_keys = all_keys - ({"text_predictions", "community_vector"} | sample_keys)
                keep_keys -= taxonomy_keys
        
        self.filtered = []
        for r in result:
            newr: Dict[str, Any] = {}
            for k in keep_keys:
                if k in r:
                    newr[k] = r[k]
            self.filtered.append(newr)
        return self.filtered
    
    def save(self, path: str | Path) -> None:
        """Save the latest filtered results to a NDJSON file.

        Note: This is a convenience method mirroring other wrappers; call
        run() first to populate filtered.
        """
        if not hasattr(self, 'filtered'):
            raise ValueError("No results to save. Call run() first.")
        with open(path, "w", encoding="utf-8") as f:
            for record in self.filtered:
                json.dump(record, f)
                f.write("\n")