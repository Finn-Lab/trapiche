import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence
import numpy as np

from .workflow import run_workflow
from .config import TaxonomyToBiomeParams, TaxonomyToVectorParams, TextToBiomeParams, TrapicheWorkflowParams, setup_logging

setup_logging(logfile=None)
logger = logging.getLogger(__name__)


class Community2vec:
    def __init__(self, model_name: str | None = None, model_version: str | None = None):
        # If not provided, default to config defaults
        if model_name is None or model_version is None:
            from .config import TaxonomyToVectorParams as _T2V
            _p = _T2V()
            self.model_name = model_name or _p.hf_model
            self.model_version = model_version or _p.model_version
        else:
            self.model_name = model_name
            self.model_version = model_version
        logger.info("Community2vec created", extra={"model_name": self.model_name, "model_version": self.model_version})

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

        logger.info(
            "Vectorising samples",
            extra={"model_name": self.model_name, "model_version": self.model_version},
        )
        self.vectorised_samples = vectorise_sample(list_of_tax_files, model_name=self.model_name, model_version=self.model_version)
        return self.vectorised_samples
    
    def save(self, path: str | Path) -> None:
        """Save the vectorised samples to a .npy file."""
        if not hasattr(self, 'vectorised_samples'):
            raise ValueError("No vectorised samples to save. Call transform() first.")
        logger.info("Saving vectorised samples", extra={"path": str(path)})
        np.save(path, self.vectorised_samples)


class TaxonomyToBiome:
    """Lightweight wrapper around deep taxonomy prediction.

    Mirrors the minimal style of other API classes. All heavy imports are
    deferred until prediction time to avoid increasing import cost.
    """

    def __init__(self, params: TaxonomyToBiomeParams | None = None):
        self.params = params or TaxonomyToBiomeParams()
        logger.info("TaxonomyToBiome created", extra={"params": self.params.__dict__ if hasattr(self.params, '__dict__') else str(self.params)})

    def predict(
        self,
        community_vectors: np.ndarray,
        constrain: Sequence[Sequence[str]] | None = None,
        params: TaxonomyToBiomeParams | None = None,
        *, model_name: str | None = None, model_version: str | None = None,
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
        from .taxonomy_prediction import predict_runs  # type: ignore

        logger.info(
            "Running taxonomy prediction | community_vectors_shape=%s | params=%s",
            getattr(community_vectors, "shape", None),
            getattr(_params, "__dict__", str(_params))
        )        
        self.results = predict_runs(
            community_vectors=community_vectors,
            constrain=constrain,
            params=_params
        )
        return self.results


    def save(self, path: str | Path) -> None:
        """Save the predictions list of dicts to ndjson file."""
        if not hasattr(self, 'results'):
            raise ValueError("No predictions to save. Call predict() first.")
        logger.info("Saving taxonomy predictions", extra={"path": str(path)})
        with open(path, "w", encoding="utf-8") as f:
            for record in self.results:
                json.dump(record, f)
                f.write("\n")


class TextToBiome:
    """Lightweight wrapper around the text classifier.

    Mirrors the minimal style of other API classes. Parameters are grouped in
    a dataclass, and heavy imports are deferred until prediction time.
    """

    def __init__(self, params: TextToBiomeParams | None = None) -> None:
        self.params = params or TextToBiomeParams()
        self.predictions_: List[List[str]] | None = None  # last predictions (optional convenience)
        logger.info("TextToBiome created", extra={"params": self.params.__dict__ if hasattr(self.params, '__dict__') else str(self.params)})

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
        logger.info("Running text prediction")
        preds = tt.predict(
            texts,
            model_name=_p.hf_model,
            model_version=_p.model_version,
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
        logger.info("Saving text predictions", extra={"path": str(path)})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.predictions_, f, indent=2)


class TrapicheWorkflowFromSequence:
    """Lightweight wrapper that runs the in-repo `run_workflow` on a sequence
    of sample dicts and returns the augmented sequence.

    The class mirrors the simple API style used elsewhere in this module and
    accepts an optional `TrapicheWorkflowParams` instance to control which
    steps run and whether intermediate results are kept.
    """

    def __init__(self, 
                 workflow_params: TrapicheWorkflowParams | None = None,
                 text_params: TextToBiomeParams | None = None,
                 vectorise_params: TaxonomyToVectorParams | None = None,
                 taxonomy_params: TaxonomyToBiomeParams | None = None,
                 ) -> None:
        self.workflow_params = workflow_params or TrapicheWorkflowParams()
        self.text_params = text_params or TextToBiomeParams()
        self.vectorise_params = vectorise_params or TaxonomyToVectorParams()
        self.taxonomy_params = taxonomy_params or TaxonomyToBiomeParams()
        

        logger.info( "TrapicheWorkflowFromSequence created",)
        logger.info("workflow_params | %s", self.workflow_params)
        logger.info("text_params | %s", self.text_params)
        logger.info("vectorise_params | %s", self.vectorise_params)
        logger.info("taxonomy_params | %s", self.taxonomy_params)

    def run(self, samples: Sequence[Dict[str, Any]], *, model_name: str | None = None, model_version: str | None = None) -> Sequence[Dict[str, Any]]:
        """Run the workflow on `samples` and return the augmented list.

        Parameters
        ----------
        samples : Sequence[Dict[str, Any]]
            Each dict should contain optional keys `project_description_file_path`
            and `taxonomy_files_paths` (list). The function returns a new list
            of dicts (shallow copies) augmented with results per sample.
        """

        result = run_workflow(
            samples,
            run_text=self.workflow_params.run_text,
            run_vectorise=self.workflow_params.run_vectorise,
            run_taxonomy=self.workflow_params.run_taxonomy,
            text_params=self.text_params,
            vectorise_params=self.vectorise_params,
            taxonomy_params=self.taxonomy_params,
            sample_study_text_heuristic=self.workflow_params.sample_study_text_heuristic,
        )

        # process which keys to keep according to config
        keep_keys = set()
        sample_keys = set(samples[0].keys()) if samples else set()
        all_keys = set(result[0].keys()) if result else set()
        if self.workflow_params.output_keys:
            keep_keys = set(self.workflow_params.output_keys)
        else:
            keep_keys = set(result[0].keys()) if result else set()
            if not self.workflow_params.keep_text_results:
                keep_keys.discard("text_predictions")
            if not self.workflow_params.keep_vectorise_results:
                keep_keys.discard("community_vector")
            if not self.workflow_params.keep_taxonomy_results:
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
        logger.info("Workflow finished", extra={"n_results": len(self.filtered)})
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