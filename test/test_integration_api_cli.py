"""Integration tests for the Trapiche API and CLI.

These tests exercise the public wrappers and the CLI end to end. Heavy
dependencies and remote Hugging Face assets are checked and tests are
skipped when unavailable to keep the suite fast and robust.
"""
import json
import os
import sys
import tempfile
import unittest
import subprocess
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
    "taxonomy_files_paths": [str(SSU), str(LSU)],
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
        self.assertIsInstance(preds[0], list)
        # Labels (if any) should be strings
        for label in preds[0]:
            self.assertIsInstance(label, str)

    def test_workflow_text_only(self):
        if not _have_module("transformers"):
            self.skipTest("transformers not installed; skipping workflow text-only integration test")
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
        self.assertIsInstance(r0["text_predictions"], list)
        self.assertEqual(r0["text_predictions"], {'root:Engineered:Built environment': 0.9994168281555176})

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
        from trapiche.config import TaxonomyToVectorParams, TaxonomyToBiomeParams
        vec_cfg = TaxonomyToVectorParams()
        tax_cfg = TaxonomyToBiomeParams()
        if not _hf_asset_available(vec_cfg.hf_model, vec_cfg.model_version, "community2vec_model_vocab_v*.json"):
            self.skipTest("HF vectorizer assets unavailable; skipping taxonomy integration test")
        if not _hf_asset_available(tax_cfg.hf_model, tax_cfg.model_version, "taxonomy_to_biome_v*.model.h5"):
            self.skipTest("HF taxonomy model asset unavailable; skipping taxonomy integration test")

        from trapiche.api import Community2vec, TaxonomyToBiome, TextToBiome

        # Text constraints
        ttb = TextToBiome()
        text_constraints = ttb.predict([SAMPLE["project_description_text"]])
        # Assert exact expected text prediction
        self.assertEqual(text_constraints, [{'root:Engineered:Built environment': 0.9994168281555176}])

        # Vectorise taxonomy files
        c2v = Community2vec()
        vectors = c2v.transform([SAMPLE["taxonomy_files_paths"]])
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
            cmd = [sys.executable, "-m", "trapiche.cli", str(in_path), "--no-vectorise", "--no-taxonomy", "--log-file", str(Path(tmpdir.name)/"trapiche.log"), "-v"]
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
            if proc.returncode != 0:
                self.fail(f"CLI failed: returncode={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}")

            # Default output file name: <input_basename>_trapiche_results.ndjson
            out_path = in_path.with_name("input_trapiche_results.ndjson")
            self.assertTrue(out_path.exists(), f"Expected output file not found: {out_path}")
            lines = out_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            rec = json.loads(lines[0])
            self.assertIn("text_predictions", rec)
            self.assertEqual(rec["text_predictions"], {'root:Engineered:Built environment': 0.9994168281555176})
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

        from trapiche.config import TaxonomyToVectorParams, TaxonomyToBiomeParams
        vec_cfg = TaxonomyToVectorParams()
        tax_cfg = TaxonomyToBiomeParams()
        if not _hf_asset_available(vec_cfg.hf_model, vec_cfg.model_version, "community2vec_model_vocab_v*.json"):
            self.skipTest("HF vectorizer assets unavailable; skipping full CLI integration test")
        if not _hf_asset_available(tax_cfg.hf_model, tax_cfg.model_version, "taxonomy_to_biome_v*.model.h5"):
            self.skipTest("HF taxonomy model asset unavailable; skipping full CLI integration test")

        tmpdir, in_path = self._write_ndjson([SAMPLE])
        try:
            cmd = [sys.executable, "-m", "trapiche.cli", str(in_path), "--log-file", str(Path(tmpdir.name)/"trapiche_full.log"), "-v"]
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
            if proc.returncode != 0:
                self.fail(f"CLI failed: returncode={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}")

            out_path = in_path.with_name("input_trapiche_results.ndjson")
            self.assertTrue(out_path.exists(), f"Expected output file not found: {out_path}")
            lines = out_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            rec = json.loads(lines[0])
            # Assert text predictions are as expected
            self.assertIn("text_predictions", rec)
            self.assertEqual(rec["text_predictions"], {'root:Engineered:Built environment': 0.9994168281555176})
            # In full mode we expect at least final_selected_prediction key when taxonomy ran successfully
            self.assertIn("final_selected_prediction", rec)
            self.assertIn("root:Engineered:Built environment", rec["final_selected_prediction"]) 
        finally:
            tmpdir.cleanup()


if __name__ == "__main__":
    unittest.main()
