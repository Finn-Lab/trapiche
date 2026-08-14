"""Unit tests for model path/version configuration (flags, env vars, config file)."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class _EnvCleanup:
    """Context manager that restores a set of env vars after the block."""

    def __init__(self, keys):
        self.keys = list(keys)
        self._saved = {}

    def __enter__(self):
        for k in self.keys:
            self._saved[k] = os.environ.get(k)
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


class TestModelConfigAliases(unittest.TestCase):
    """Model-selection fields must not collide across the three params classes."""

    ENV_KEYS = [
        "TRAPICHE_HF_MODEL",
        "TRAPICHE_MODEL_VERSION",
        "TRAPICHE_LOCAL_MODEL_DIR",
        "TRAPICHE_TEXT_HF_MODEL",
        "TRAPICHE_TEXT_MODEL_VERSION",
        "TRAPICHE_TEXT_LOCAL_MODEL_DIR",
        "TRAPICHE_VECTOR_HF_MODEL",
        "TRAPICHE_VECTOR_MODEL_VERSION",
        "TRAPICHE_VECTOR_LOCAL_MODEL_DIR",
        "TRAPICHE_TAXONOMY_HF_MODEL",
        "TRAPICHE_TAXONOMY_MODEL_VERSION",
        "TRAPICHE_TAXONOMY_LOCAL_MODEL_DIR",
    ]

    def test_distinct_env_vars_do_not_collide(self):
        from trapiche.config import TaxonomyToBiomeParams, TaxonomyToVectorParams, TextToBiomeParams

        with _EnvCleanup(self.ENV_KEYS):
            os.environ["TRAPICHE_TEXT_HF_MODEL"] = "text-repo"
            os.environ["TRAPICHE_VECTOR_HF_MODEL"] = "vector-repo"
            os.environ["TRAPICHE_TAXONOMY_HF_MODEL"] = "taxonomy-repo"

            self.assertEqual(TextToBiomeParams().hf_model, "text-repo")
            self.assertEqual(TaxonomyToVectorParams().hf_model, "vector-repo")
            self.assertEqual(TaxonomyToBiomeParams().hf_model, "taxonomy-repo")

    def test_legacy_broadcast_env_var_still_works_as_fallback(self):
        from trapiche.config import TaxonomyToVectorParams, TextToBiomeParams

        with _EnvCleanup(self.ENV_KEYS):
            os.environ["TRAPICHE_HF_MODEL"] = "legacy-repo"
            self.assertEqual(TextToBiomeParams().hf_model, "legacy-repo")
            self.assertEqual(TaxonomyToVectorParams().hf_model, "legacy-repo")

    def test_specific_env_var_takes_precedence_over_legacy(self):
        from trapiche.config import TextToBiomeParams

        with _EnvCleanup(self.ENV_KEYS):
            os.environ["TRAPICHE_HF_MODEL"] = "legacy-repo"
            os.environ["TRAPICHE_TEXT_HF_MODEL"] = "specific-repo"
            self.assertEqual(TextToBiomeParams().hf_model, "specific-repo")

    def test_local_model_dir_defaults_to_none(self):
        from trapiche.config import TaxonomyToBiomeParams, TaxonomyToVectorParams, TextToBiomeParams

        with _EnvCleanup(self.ENV_KEYS):
            self.assertIsNone(TextToBiomeParams().local_model_dir)
            self.assertIsNone(TaxonomyToVectorParams().local_model_dir)
            self.assertIsNone(TaxonomyToBiomeParams().local_model_dir)

    def test_local_model_dir_env_var_per_class(self):
        from trapiche.config import TaxonomyToBiomeParams, TaxonomyToVectorParams, TextToBiomeParams

        with _EnvCleanup(self.ENV_KEYS):
            os.environ["TRAPICHE_TEXT_LOCAL_MODEL_DIR"] = "/models/text"
            os.environ["TRAPICHE_VECTOR_LOCAL_MODEL_DIR"] = "/models/vector"
            os.environ["TRAPICHE_TAXONOMY_LOCAL_MODEL_DIR"] = "/models/taxonomy"

            self.assertEqual(TextToBiomeParams().local_model_dir, "/models/text")
            self.assertEqual(TaxonomyToVectorParams().local_model_dir, "/models/vector")
            self.assertEqual(TaxonomyToBiomeParams().local_model_dir, "/models/taxonomy")

    def test_fields_still_settable_by_keyword(self):
        from trapiche.config import TextToBiomeParams

        with _EnvCleanup(self.ENV_KEYS):
            p = TextToBiomeParams(hf_model="by-kwarg", model_version="2.0")
            self.assertEqual(p.hf_model, "by-kwarg")
            self.assertEqual(p.model_version, "2.0")


class TestLoadConfigFile(unittest.TestCase):
    def test_load_config_file_sets_env_vars(self):
        from trapiche.config import load_config_file

        with _EnvCleanup(TestModelConfigAliases.ENV_KEYS), tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "trapiche.env"
            config_path.write_text(
                "TRAPICHE_TEXT_HF_MODEL=from-file-repo\n" "TRAPICHE_TEXT_MODEL_VERSION=2.5\n"
            )
            load_config_file(config_path)
            self.assertEqual(os.environ.get("TRAPICHE_TEXT_HF_MODEL"), "from-file-repo")
            self.assertEqual(os.environ.get("TRAPICHE_TEXT_MODEL_VERSION"), "2.5")

    def test_load_config_file_missing_raises(self):
        from trapiche.config import load_config_file

        with self.assertRaises(FileNotFoundError):
            load_config_file("/nonexistent/path/trapiche.env")


class TestGetHfModelPathLocalOverride(unittest.TestCase):
    def test_local_model_dir_resolves_existing_file(self):
        from trapiche.utils import _get_hf_model_path

        with tempfile.TemporaryDirectory() as tmp:
            version_dir = Path(tmp) / "1.0"
            version_dir.mkdir()
            asset = version_dir / "vocab_1.0.txt"
            asset.write_text("hello")

            resolved = _get_hf_model_path("ignored/repo", "1.0", "vocab_*.txt", tmp)
            self.assertEqual(Path(resolved), asset)

    def test_local_model_dir_missing_file_raises(self):
        from trapiche.utils import _get_hf_model_path

        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(FileNotFoundError):
            _get_hf_model_path("ignored/repo", "1.0", "vocab_*.txt", tmp)

    def test_no_local_model_dir_falls_back_to_hf_hub_download(self):
        from trapiche.utils import _get_hf_model_path

        with patch("trapiche.utils.hf_hub_download", return_value="/cache/path/vocab_1.0.txt") as m:
            resolved = _get_hf_model_path("some/repo", "1.0", "vocab_*.txt")
            self.assertEqual(str(resolved), "/cache/path/vocab_1.0.txt")
            m.assert_called_once_with(
                repo_id="some/repo", filename="1.0/vocab_1.0.txt", repo_type="model"
            )


class TestCliModelFlags(unittest.TestCase):
    ENV_KEYS = TestModelConfigAliases.ENV_KEYS

    def test_parse_args_recognizes_model_flags(self):
        from trapiche.cli import parse_args

        args = parse_args(
            [
                "input.ndjson",
                "--text-hf-model",
                "org/text-model",
                "--text-model-version",
                "3.0",
                "--text-local-model-dir",
                "/models/text",
                "--vector-hf-model",
                "org/vector-model",
                "--taxonomy-local-model-dir",
                "/models/taxonomy",
                "--config",
                "config.env",
            ]
        )
        self.assertEqual(args.text_hf_model, "org/text-model")
        self.assertEqual(args.text_model_version, "3.0")
        self.assertEqual(args.text_local_model_dir, "/models/text")
        self.assertEqual(args.vector_hf_model, "org/vector-model")
        self.assertEqual(args.taxonomy_local_model_dir, "/models/taxonomy")
        self.assertEqual(args.config_file, "config.env")

    def test_main_threads_cli_flags_into_model_params(self):
        from trapiche import cli

        with _EnvCleanup(self.ENV_KEYS), tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.ndjson"
            input_path.write_text(json.dumps({"sample_id": "s1"}) + "\n")
            log_path = Path(tmp) / "trapiche.log"

            mock_instance = MagicMock()
            mock_instance.run.return_value = []
            mock_runner_cls = MagicMock(return_value=mock_instance)

            with patch.object(cli, "TrapicheWorkflowFromSequence", mock_runner_cls):
                rc = cli.main(
                    [
                        str(input_path),
                        "-o",
                        str(Path(tmp) / "out.ndjson"),
                        "--log-file",
                        str(log_path),
                        "--text-hf-model",
                        "org/custom-text",
                        "--text-model-version",
                        "9.9",
                        "--taxonomy-local-model-dir",
                        "/models/taxonomy",
                    ]
                )

            self.assertEqual(rc, 0)
            mock_runner_cls.assert_called_once()
            _, kwargs = mock_runner_cls.call_args
            self.assertEqual(kwargs["text_params"].hf_model, "org/custom-text")
            self.assertEqual(kwargs["text_params"].model_version, "9.9")
            self.assertEqual(kwargs["taxonomy_params"].local_model_dir, "/models/taxonomy")
            # Env vars are also set, so any lazily-reinstantiated config
            # elsewhere in the process picks up the same override.
            self.assertEqual(os.environ.get("TRAPICHE_TEXT_HF_MODEL"), "org/custom-text")
            self.assertEqual(
                os.environ.get("TRAPICHE_TAXONOMY_LOCAL_MODEL_DIR"), "/models/taxonomy"
            )

    def test_config_file_applied_and_overridden_by_explicit_flag(self):
        from trapiche import cli

        with _EnvCleanup(self.ENV_KEYS), tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.ndjson"
            input_path.write_text(json.dumps({"sample_id": "s1"}) + "\n")
            log_path = Path(tmp) / "trapiche.log"
            config_path = Path(tmp) / "trapiche.env"
            config_path.write_text(
                "TRAPICHE_VECTOR_HF_MODEL=from-file-repo\n"
                "TRAPICHE_TEXT_HF_MODEL=from-file-text-repo\n"
            )

            mock_instance = MagicMock()
            mock_instance.run.return_value = []
            mock_runner_cls = MagicMock(return_value=mock_instance)

            with patch.object(cli, "TrapicheWorkflowFromSequence", mock_runner_cls):
                cli.main(
                    [
                        str(input_path),
                        "-o",
                        str(Path(tmp) / "out.ndjson"),
                        "--log-file",
                        str(log_path),
                        "--config",
                        str(config_path),
                        # explicit flag should win over the config file value
                        "--text-hf-model",
                        "org/explicit-text",
                    ]
                )

            _, kwargs = mock_runner_cls.call_args
            # Value only present in the config file is applied.
            self.assertEqual(kwargs["vectorise_params"].hf_model, "from-file-repo")
            # Explicit CLI flag overrides the config file value.
            self.assertEqual(kwargs["text_params"].hf_model, "org/explicit-text")


class TestTaxonomyModelResolution(unittest.TestCase):
    def test_explicit_taxonomy_params_select_the_model_file(self):
        from trapiche.config import TaxonomyToBiomeParams
        from trapiche.taxonomy_prediction import _resolve_model_file

        params = TaxonomyToBiomeParams(
            hf_model="org/custom-taxonomy", model_version="2.0", local_model_dir="/models/taxonomy"
        )
        with patch(
            "trapiche.taxonomy_prediction._get_hf_model_path", return_value="/models/model.h5"
        ) as get_path:
            self.assertEqual(_resolve_model_file(params), "/models/model.h5")

        get_path.assert_called_once_with(
            "org/custom-taxonomy",
            "2.0",
            "taxonomy_to_biome_v*.model.h5",
            "/models/taxonomy",
        )


if __name__ == "__main__":
    unittest.main()
