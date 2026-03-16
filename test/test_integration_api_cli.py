"""Integration tests for the Trapiche API and CLI.

These tests exercise the public wrappers and the CLI end to end. Heavy
dependencies and remote Hugging Face assets are checked and tests are
skipped when unavailable to keep the suite fast and robust.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Reuse local test taxonomy files
REPO_ROOT = Path(__file__).resolve().parents[1]
TAX_DIR = REPO_ROOT / "test" / "taxonomy_files"
SSU = TAX_DIR / "SRR1524511_MERGED_FASTQ_SSU_OTU.tsv"
LSU = TAX_DIR / "SRR1524511_MERGED_FASTQ_LSU_OTU.tsv"

SAMPLE = {
    "project_description_text": (
        "Home Microbiome Metagenomes. The project identifies patterns in microbial "
        "communities associated with different home and home occupant (human and pet) surfaces"
    ),
    "sample_description_text": (
        "Metagenome of microbial community: Bedroom Floor. "
        "House_04a-Bedroom_Floor_Day3. House_04a-Bedroom_Floor_Day3"
    ),
    "sample_taxonomy_paths": [str(SSU), str(LSU)],
}


def _have_module(mod_name: str) -> bool:
    try:
        __import__(mod_name)
        return True
    except Exception:
        return False


def _hf_asset_available(model_name: str, model_version: str, pattern: str) -> bool:
    """Best-effort check for a single HF asset; returns False if download fails."""
    try:
        from trapiche.utils import _get_hf_model_path  # local import

        p = _get_hf_model_path(model_name, model_version, pattern)
        return Path(p).exists()
    except Exception:
        return False


class TestAPIIntegration(unittest.TestCase):
    def test_text_api_predict(self):
        if not _have_module("transformers"):
            self.skipTest("transformers not installed; skipping text prediction integration test")
        from trapiche.api import TextToBiome

        ttb = TextToBiome()  # defaults from config
        texts = [SAMPLE["project_description_text"]]
        preds = ttb.predict(texts)
        self.assertIsInstance(preds, list)
        self.assertEqual(len(preds), 1)
        self.assertIsInstance(preds[0], dict)
        # Labels (if any) should be strings
        for label in preds[0]:
            self.assertIsInstance(label, str)

    def test_workflow_text_only(self):
        if not _have_module("transformers"):
            self.skipTest(
                "transformers not installed; skipping workflow text-only integration test"
            )
        from trapiche.api import TrapicheWorkflowFromSequence
        from trapiche.config import TrapicheWorkflowParams

        params = TrapicheWorkflowParams(
            run_text=True,
            run_vectorise=False,
            run_taxonomy=False,
            # Keep output compact but ensure text_predictions is included by allowing defaults
        )
        runner = TrapicheWorkflowFromSequence(workflow_params=params)
        res = runner.run([SAMPLE])
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 1)
        r0 = res[0]
        # In text-only mode we expect exact text_predictions
        self.assertIn("text_predictions", r0)
        self.assertIsInstance(r0["text_predictions"], dict)
        self.assertEqual(
            r0["text_predictions"], {"root:Engineered:Built environment": 0.9994168281555176}
        )

    def test_vector_and_taxonomy_api(self):
        # Require optional heavy deps: tensorflow, pandas, tables, gensim
        if not _have_module("tensorflow"):
            self.skipTest("tensorflow not installed; skipping taxonomy integration test")
        if not _have_module("pandas"):
            self.skipTest("pandas not installed; skipping taxonomy integration test")
        if not _have_module("tables") and not _have_module("pytables"):
            self.skipTest("PyTables not installed; skipping taxonomy integration test")
        if not _have_module("gensim"):
            self.skipTest("gensim not installed; skipping taxonomy integration test")

        # Check HF assets best-effort; skip if not accessible
        from trapiche.config import TaxonomyToBiomeParams, TaxonomyToVectorParams

        vec_cfg = TaxonomyToVectorParams()
        tax_cfg = TaxonomyToBiomeParams()
        if not _hf_asset_available(
            vec_cfg.hf_model, vec_cfg.model_version, "community2vec_model_vocab_v*.json"
        ):
            self.skipTest("HF vectorizer assets unavailable; skipping taxonomy integration test")
        if not _hf_asset_available(
            tax_cfg.hf_model, tax_cfg.model_version, "taxonomy_to_biome_v*.model.h5"
        ):
            self.skipTest("HF taxonomy model asset unavailable; skipping taxonomy integration test")

        from trapiche.api import Community2vec, TaxonomyToBiome, TextToBiome

        # Text constraints
        ttb = TextToBiome()
        text_constraints = ttb.predict([SAMPLE["project_description_text"]])
        # Assert exact expected text prediction
        self.assertEqual(
            text_constraints, [{"root:Engineered:Built environment": 0.9994168281555176}]
        )

        # Vectorise taxonomy files
        c2v = Community2vec()
        vectors = c2v.transform([SAMPLE])
        self.assertEqual(vectors.shape[0], 1)
        self.assertGreaterEqual(vectors.shape[1], 0)

        # Predict taxonomy with constraints
        tax2b = TaxonomyToBiome()
        results = tax2b.predict(community_vectors=vectors, constrain=text_constraints)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        r0 = results[0]
        self.assertIsInstance(r0, dict)
        self.assertIn("final_selected_prediction", r0)
        # Assert the final prediction matches expected label
        final_pred = r0["final_selected_prediction"]
        self.assertIsInstance(final_pred, dict)
        self.assertIn("root:Engineered:Built environment", final_pred)


class TestCLIIntegration(unittest.TestCase):
    def _write_ndjson(self, records):
        tmpdir = tempfile.TemporaryDirectory()
        in_path = Path(tmpdir.name) / "input.ndjson"
        with open(in_path, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r))
                fh.write("\n")
        return tmpdir, in_path

    def test_cli_text_only(self):
        if not _have_module("transformers"):
            self.skipTest("transformers not installed; skipping CLI text-only test")
        tmpdir, in_path = self._write_ndjson([SAMPLE])
        try:
            # Let CLI derive default output path based on input basename
            cmd = [
                sys.executable,
                "-m",
                "trapiche.cli",
                str(in_path),
                "--no-run-vectorise",
                "--no-run-taxonomy",
                "--log-file",
                str(Path(tmpdir.name) / "trapiche.log"),
                "-v",
            ]
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
            if proc.returncode != 0:
                self.fail(
                    f"CLI failed: returncode={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
                )

            # Default output file name: <input_basename>_trapiche_results.ndjson
            out_path = in_path.with_name("input_trapiche_results.ndjson")
            self.assertTrue(out_path.exists(), f"Expected output file not found: {out_path}")
            lines = out_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            rec = json.loads(lines[0])
            self.assertIn("text_predictions", rec)
            self.assertEqual(
                rec["text_predictions"], {"root:Engineered:Built environment": 0.9994168281555176}
            )
        finally:
            tmpdir.cleanup()

    def test_cli_full_if_available(self):
        # Only attempt full run if heavy deps and assets are present
        if not _have_module("tensorflow"):
            self.skipTest("tensorflow not installed; skipping full CLI integration test")
        if not _have_module("pandas"):
            self.skipTest("pandas not installed; skipping full CLI integration test")
        if not _have_module("tables") and not _have_module("pytables"):
            self.skipTest("PyTables not installed; skipping full CLI integration test")
        if not _have_module("gensim"):
            self.skipTest("gensim not installed; skipping full CLI integration test")

        from trapiche.config import TaxonomyToBiomeParams, TaxonomyToVectorParams

        vec_cfg = TaxonomyToVectorParams()
        tax_cfg = TaxonomyToBiomeParams()
        if not _hf_asset_available(
            vec_cfg.hf_model, vec_cfg.model_version, "community2vec_model_vocab_v*.json"
        ):
            self.skipTest("HF vectorizer assets unavailable; skipping full CLI integration test")
        if not _hf_asset_available(
            tax_cfg.hf_model, tax_cfg.model_version, "taxonomy_to_biome_v*.model.h5"
        ):
            self.skipTest("HF taxonomy model asset unavailable; skipping full CLI integration test")

        tmpdir, in_path = self._write_ndjson([SAMPLE])
        try:
            cmd = [
                sys.executable,
                "-m",
                "trapiche.cli",
                str(in_path),
                "--log-file",
                str(Path(tmpdir.name) / "trapiche_full.log"),
                "-v",
            ]
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
            if proc.returncode != 0:
                self.fail(
                    f"CLI failed: returncode={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
                )

            out_path = in_path.with_name("input_trapiche_results.ndjson")
            self.assertTrue(out_path.exists(), f"Expected output file not found: {out_path}")
            lines = out_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            rec = json.loads(lines[0])
            # Assert text predictions are as expected
            self.assertIn("text_predictions", rec)
            self.assertEqual(
                rec["text_predictions"], {"root:Engineered:Built environment": 0.9994168281555176}
            )
            # In full mode we expect at least final_selected_prediction key when taxonomy ran successfully
            self.assertIn("final_selected_prediction", rec)
            self.assertIn("root:Engineered:Built environment", rec["final_selected_prediction"])
        finally:
            tmpdir.cleanup()


class TestExternalTextPredictions(unittest.TestCase):
    """Tests for the external-label short-circuit in run_text_step.

    These tests do not require model downloads — the internal BERT model is
    never called when external keys are present.
    """

    def _run(self, samples, use_heuristic=False):
        from trapiche.config import TextToBiomeParams
        from trapiche.workflow import run_text_step

        params = TextToBiomeParams()
        return run_text_step(samples, params_obj=params, use_heuristic=use_heuristic)

    def test_external_project_only(self):
        """Single sample with ext_text_pred_project → correct text_predictions."""
        samples = [{"ext_text_pred_project": ["root:Environmental:Aquatic"]}]
        combined, proj, samp, flags, raw_proj, raw_samp = self._run(samples)
        self.assertEqual(combined[0], {"root:Environmental:Aquatic": 1.0})
        self.assertEqual(proj[0], {"root:Environmental:Aquatic": 1.0})
        self.assertIsNone(samp[0])
        self.assertFalse(flags[0])

    def test_external_heuristic(self):
        """Both ext keys + use_heuristic=True → heuristic applied."""
        samples = [
            {
                "ext_text_pred_project": ["root:Environmental:Aquatic:Marine"],
                "ext_text_pred_sample": ["root:Environmental:Aquatic"],
            }
        ]
        combined, proj, samp, flags, raw_proj, raw_samp = self._run(samples, use_heuristic=True)
        # heuristic should fire
        self.assertTrue(flags[0])
        self.assertIsNotNone(combined[0])
        self.assertIsNotNone(proj[0])
        self.assertIsNotNone(samp[0])

    def test_mixed_batch_skips_internal(self):
        """Batch where only one sample has ext key → other sample gets None; internal model never called."""
        from unittest.mock import patch

        samples = [
            {"ext_text_pred_project": ["root:Environmental:Aquatic"]},
            {"project_description_text": "some text"},
        ]
        with patch("trapiche.workflow.tt") as mock_tt:
            combined, proj, samp, flags, raw_proj, raw_samp = self._run(samples)
            mock_tt.predict.assert_not_called()

        self.assertEqual(combined[0], {"root:Environmental:Aquatic": 1.0})
        self.assertIsNone(combined[1])

    def test_invalid_label_raises(self):
        """A malformed label raises ValueError."""
        from trapiche.utils import validate_biome_labels

        with self.assertRaises(ValueError):
            validate_biome_labels(["not-a-valid-biome"])

        with self.assertRaises(ValueError):
            validate_biome_labels(["root"])  # no child nodes

    def test_empty_list_is_none(self):
        """ext_text_pred_project=[] is treated as absent → combined is None."""
        samples = [{"ext_text_pred_project": []}]
        combined, proj, samp, flags, raw_proj, raw_samp = self._run(samples)
        self.assertIsNone(combined[0])
        self.assertIsNone(proj[0])
        self.assertFalse(flags[0])


class TestBiomeLabelNormalization(unittest.TestCase):
    """Tests for normalize_and_canonicalize_labels and helpers."""

    def _skip_if_no_asset(self):
        try:
            from trapiche.utils import load_biome_herarchy_dict

            load_biome_herarchy_dict()
        except FileNotFoundError:
            self.skipTest("Biome hierarchy asset not available")

    def test_normalize_label_str(self):
        from trapiche.utils import _normalize_label_str

        self.assertEqual(
            _normalize_label_str(" root : Environmental : Aquatic "),
            "root:environmental:aquatic",
        )

    def test_canonicalize_known_label(self):
        """Exact-match lowercase label resolves to canonical form."""
        self._skip_if_no_asset()
        from trapiche.utils import normalize_and_canonicalize_labels

        result = normalize_and_canonicalize_labels(["root:environmental:aquatic"])
        self.assertEqual(result, ["root:Environmental:Aquatic"])

    def test_canonicalize_label_with_spaces(self):
        """Label with surrounding spaces/mixed case resolves to canonical form."""
        self._skip_if_no_asset()
        from trapiche.utils import normalize_and_canonicalize_labels

        result = normalize_and_canonicalize_labels(["root:Environmental :Aquatic"])
        self.assertEqual(result, ["root:Environmental:Aquatic"])

    def test_unknown_label_dropped(self):
        """A label not in the GOLD ontology is dropped and a warning is emitted."""
        self._skip_if_no_asset()
        import logging

        from trapiche.utils import normalize_and_canonicalize_labels

        with self.assertLogs("trapiche.utils", level=logging.WARNING):
            result = normalize_and_canonicalize_labels(["root:Nonexistent:Biome"])
        self.assertEqual(result, [])

    def test_mixed_labels(self):
        """One valid + one unknown → only the valid canonical label is returned."""
        self._skip_if_no_asset()
        from trapiche.utils import normalize_and_canonicalize_labels

        result = normalize_and_canonicalize_labels(
            ["root:environmental:aquatic", "root:Nonexistent:Biome"]
        )
        self.assertEqual(result, ["root:Environmental:Aquatic"])

    def test_to_trapiche_samples_normalizes(self):
        """to_trapiche_samples canonicalizes labels with extra whitespace."""
        self._skip_if_no_asset()
        from trapiche.helpers.llm_text_pred import to_trapiche_samples

        enriched = [
            {
                "project_id": "p1",
                "project_ecosystems": ["root:Environmental :Aquatic:Marine "],
                "samples": [
                    {
                        "sample_id": "s1",
                        "sample_ecosystems": [" root:Environmental:Aquatic"],
                    }
                ],
            }
        ]
        result = to_trapiche_samples(enriched)
        self.assertEqual(len(result), 1)
        self.assertIn("root:Environmental:Aquatic:Marine", result[0]["ext_text_pred_project"])
        self.assertIn("root:Environmental:Aquatic", result[0]["ext_text_pred_sample"])

    def test_external_step_normalizes(self):
        """_run_text_step_external accepts lowercase labels and returns canonical predictions."""
        self._skip_if_no_asset()
        from trapiche.config import TextToBiomeParams
        from trapiche.workflow import run_text_step

        samples = [{"ext_text_pred_project": ["root:environmental:aquatic"]}]
        params = TextToBiomeParams()
        combined, proj, samp, flags, raw_proj, raw_samp = run_text_step(
            samples, params_obj=params, use_heuristic=False
        )
        self.assertIsNotNone(combined[0])
        self.assertIn("root:Environmental:Aquatic", combined[0])

    def test_fuzzy_match_label_exact_overlap(self):
        """_fuzzy_match_label returns expected canonical for partial-overlap label."""
        from trapiche.utils import _fuzzy_match_label

        lower_to_canonical = {
            "root:environmental:aquatic": "root:Environmental:Aquatic",
            "root:environmental:terrestrial": "root:Environmental:Terrestrial",
            "root:engineered": "root:Engineered",
        }
        # "root:environmental:aquatic:marine" shares 3 terms with aquatic key
        result = _fuzzy_match_label("root:environmental:aquatic:marine", lower_to_canonical)
        self.assertEqual(result, ["root:Environmental:Aquatic"])

    def test_fuzzy_match_label_tie(self):
        """_fuzzy_match_label returns all tied candidates."""
        from trapiche.utils import _fuzzy_match_label

        lower_to_canonical = {
            "root:a:b": "root:A:B",
            "root:a:c": "root:A:C",
        }
        # "root:a:x" shares 2 terms with both keys
        result = _fuzzy_match_label("root:a:x", lower_to_canonical)
        self.assertIn("root:A:B", result)
        self.assertIn("root:A:C", result)
        self.assertEqual(len(result), 2)

    def test_fuzzy_match_label_no_overlap(self):
        """_fuzzy_match_label returns [] when there is no term overlap."""
        from trapiche.utils import _fuzzy_match_label

        lower_to_canonical = {"root:environmental:aquatic": "root:Environmental:Aquatic"}
        result = _fuzzy_match_label("completely:different:terms", lower_to_canonical)
        self.assertEqual(result, [])

    def test_normalize_with_fuzzy_fallback(self):
        """normalize_and_canonicalize_labels with fuzzy_fallback=True resolves near-miss."""
        self._skip_if_no_asset()
        from trapiche.utils import normalize_and_canonicalize_labels

        # "root:Environmental:Aquatic:Marine" may or may not be in the ontology;
        # "root:Environmental:Aquatic" is known. A near-miss unknown label that
        # overlaps maximally should resolve rather than being dropped.
        # Use a mock lower_to_canonical to test the fuzzy path directly.
        from unittest.mock import patch
        from trapiche import utils as _utils

        fake_canonical = {
            "root:environmental:aquatic": "root:Environmental:Aquatic",
        }
        with patch.object(_utils, "_build_lower_to_canonical", return_value=fake_canonical):
            # exact match — should still work
            result_exact = normalize_and_canonicalize_labels(
                ["root:Environmental:Aquatic"], fuzzy_fallback=True
            )
            self.assertEqual(result_exact, ["root:Environmental:Aquatic"])

            # near-miss: shares root+environmental+aquatic (3 terms)
            result_fuzzy = normalize_and_canonicalize_labels(
                ["root:Environmental:Aquatic:Marine"], fuzzy_fallback=True
            )
            self.assertEqual(result_fuzzy, ["root:Environmental:Aquatic"])

    def test_normalize_without_fuzzy_fallback_drops(self):
        """normalize_and_canonicalize_labels without fuzzy_fallback drops unknown labels."""
        self._skip_if_no_asset()
        from unittest.mock import patch
        from trapiche import utils as _utils
        from trapiche.utils import normalize_and_canonicalize_labels

        fake_canonical = {
            "root:environmental:aquatic": "root:Environmental:Aquatic",
        }
        with patch.object(_utils, "_build_lower_to_canonical", return_value=fake_canonical):
            import logging

            with self.assertLogs("trapiche.utils", level=logging.WARNING):
                result = normalize_and_canonicalize_labels(
                    ["root:Environmental:Aquatic:Marine"], fuzzy_fallback=False
                )
            self.assertEqual(result, [])


class TestExternalRawLabelPreservation(unittest.TestCase):
    """Tests for raw label preservation in the external pathway."""

    def _skip_if_no_asset(self):
        try:
            from trapiche.utils import load_biome_herarchy_dict

            load_biome_herarchy_dict()
        except FileNotFoundError:
            self.skipTest("Biome hierarchy asset not available")

    def test_run_text_step_returns_six_tuple_internal(self):
        """Internal BERT pathway returns 6-tuple with None raw lists."""
        from unittest.mock import MagicMock, patch
        from trapiche.config import TextToBiomeParams
        from trapiche.workflow import run_text_step

        params = TextToBiomeParams()
        samples = [{"project_description_text": "some text about the environment"}]

        mock_pred = {"root:Environmental:Aquatic": 0.9}
        with patch("trapiche.workflow.tt") as mock_tt:
            mock_tt.predict.return_value = [mock_pred]
            result = run_text_step(samples, params_obj=params, use_heuristic=False)

        self.assertEqual(len(result), 6)
        combined, proj, samp, flags, raw_proj, raw_samp = result
        self.assertIsNone(raw_proj[0])
        self.assertIsNone(raw_samp[0])

    def test_run_text_step_returns_raw_labels_external(self):
        """External pathway: raw labels are captured before canonicalization."""
        from trapiche.config import TextToBiomeParams
        from trapiche.workflow import run_text_step

        raw_label = "root:Environmental:Aquatic"
        samples = [{"ext_text_pred_project": [raw_label]}]
        params = TextToBiomeParams()
        combined, proj, samp, flags, raw_proj, raw_samp = run_text_step(
            samples, params_obj=params, use_heuristic=False
        )
        self.assertIsNotNone(raw_proj[0])
        self.assertIn(raw_label, raw_proj[0])
        self.assertIsNone(raw_samp[0])

    def test_run_workflow_emits_raw_keys(self):
        """run_workflow includes _raw_ext_text_pred_project/sample in output."""
        from unittest.mock import patch
        from trapiche.config import (
            TaxonomyToBiomeParams,
            TaxonomyToVectorParams,
            TextToBiomeParams,
        )
        from trapiche.workflow import run_workflow

        raw_label = "root:Environmental:Aquatic"
        samples = [{"ext_text_pred_project": [raw_label]}]

        with patch("trapiche.workflow.c2v_mod") as mock_c2v, patch(
            "trapiche.workflow.taxonomy_prediction"
        ):
            import numpy as np

            mock_c2v.vectorise_samples.return_value = [np.zeros(128)]
            result = run_workflow(
                samples,
                text_params=TextToBiomeParams(),
                vectorise_params=TaxonomyToVectorParams(),
                taxonomy_params=TaxonomyToBiomeParams(),
                run_text=True,
                run_vectorise=False,
                run_taxonomy=False,
            )

        self.assertEqual(len(result), 1)
        self.assertIn("_raw_ext_text_pred_project", result[0])
        self.assertIn(raw_label, result[0]["_raw_ext_text_pred_project"])

    def test_to_trapiche_samples_raw_keys(self):
        """to_trapiche_samples propagates _raw_ext_text_pred_* keys."""
        from trapiche.helpers.llm_text_pred import to_trapiche_samples

        enriched = [
            {
                "project_id": "p1",
                "project_ecosystems": ["root:Environmental:Aquatic"],
                "_raw_project_ecosystems": ["root:Environmental:Aquatic:Marine"],
                "samples": [
                    {
                        "sample_id": "s1",
                        "sample_ecosystems": ["root:Environmental:Aquatic"],
                        "_raw_sample_ecosystems": ["root:Environmental :Aquatic"],
                    }
                ],
            }
        ]
        result = to_trapiche_samples(enriched)
        self.assertEqual(len(result), 1)
        self.assertIn("_raw_ext_text_pred_project", result[0])
        self.assertIn("_raw_ext_text_pred_sample", result[0])
        self.assertIn("root:Environmental:Aquatic:Marine", result[0]["_raw_ext_text_pred_project"])
        self.assertIn("root:Environmental :Aquatic", result[0]["_raw_ext_text_pred_sample"])

    def test_validate_and_clean_saves_raw(self):
        """_validate_and_clean stores _raw_project_ecosystems and _raw_sample_ecosystems."""
        from trapiche.helpers.llm_text_pred import _validate_and_clean

        enriched = [
            {
                "project_id": "p1",
                "project_ecosystems": ["root:Environmental:Aquatic"],
                "samples": [
                    {
                        "sample_id": "s1",
                        "sample_ecosystems": ["root:Environmental:Aquatic:Marine"],
                    }
                ],
            }
        ]
        cleaned = _validate_and_clean(enriched)
        self.assertEqual(len(cleaned), 1)
        proj = cleaned[0]
        self.assertIn("_raw_project_ecosystems", proj)
        self.assertEqual(proj["_raw_project_ecosystems"], ["root:Environmental:Aquatic"])
        sample = proj["samples"][0]
        self.assertIn("_raw_sample_ecosystems", sample)
        self.assertEqual(sample["_raw_sample_ecosystems"], ["root:Environmental:Aquatic:Marine"])


if __name__ == "__main__":
    unittest.main()
