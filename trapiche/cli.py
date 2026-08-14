"""Command-line interface for running the Trapiche workflow.

Reads NDJSON input (file or stdin), executes selected steps, and writes
NDJSON output (file or stdout). Supports gzip input/output.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .api import TrapicheWorkflowFromSequence
from .config import (
    TaxonomyToBiomeParams,
    TaxonomyToVectorParams,
    TextToBiomeParams,
    TrapicheWorkflowParams,
    load_config_file,
    setup_logging,
)

# Maps CLI dest -> env var set when the flag is provided, so that CLI
# overrides are visible both to the params objects constructed here and to
# any component that lazily re-reads model config from the environment
# (e.g. the taxonomy classifier's internal model loader).
_MODEL_ENV_VARS: dict[str, str] = {
    "text_hf_model": "TRAPICHE_TEXT_HF_MODEL",
    "text_model_version": "TRAPICHE_TEXT_MODEL_VERSION",
    "text_local_model_dir": "TRAPICHE_TEXT_LOCAL_MODEL_DIR",
    "vector_hf_model": "TRAPICHE_VECTOR_HF_MODEL",
    "vector_model_version": "TRAPICHE_VECTOR_MODEL_VERSION",
    "vector_local_model_dir": "TRAPICHE_VECTOR_LOCAL_MODEL_DIR",
    "taxonomy_hf_model": "TRAPICHE_TAXONOMY_HF_MODEL",
    "taxonomy_model_version": "TRAPICHE_TAXONOMY_MODEL_VERSION",
    "taxonomy_local_model_dir": "TRAPICHE_TAXONOMY_LOCAL_MODEL_DIR",
}


def read_ndjson(path: Path | None) -> Iterable[dict[str, Any]]:
    """Yield JSON objects from NDJSON input.

    Args:
        path: Input file path or None to read from stdin. .gz supported.

    Yields:
        dict: One object per line.
    """
    try:
        if path is None:
            # stdin: text stream
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
            return

        if path.suffix == ".gz":
            stream = gzip.open(path, "rt", encoding="utf-8")
        else:
            stream = path.open(encoding="utf-8")

        with stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON encountered: {e}") from e
    except FileNotFoundError:
        raise SystemExit(f"Input file not found: {path}") from None


def write_ndjson(records: Iterable[dict[str, Any]], path: Path | None) -> None:
    """Write records as NDJSON to a file or stdout.

    Args:
        records: Iterable of dicts to serialize.
        path: Output path or None to write to stdout. .gz supported.
    """
    if path is None:
        out = sys.stdout
        for r in records:
            out.write(json.dumps(r, ensure_ascii=False))
            out.write("\n")
        out.flush()
        return

    # ensure parent exists
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False))
                fh.write("\n")
    else:
        with open(path, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False))
                fh.write("\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="trapiche-cli",
        description="Run Trapiche workflow on a sequence of sample dicts provided as NDJSON.",
    )
    p.add_argument(
        "input", nargs="?", help="Input NDJSON file path (use - for stdin). Supports .gz"
    )
    p.add_argument(
        "-o",
        "--output",
        help="Output NDJSON file path (defaults to <INPUT_BASENAME>_trapiche_results.ndjson). Use .gz to compress",
    )
    p.add_argument(
        "--disable-minimal-result",
        dest="disable_minimal_result",
        action="store_true",
        help=(
            "When set, disable the default minimal output."
            " When disabled, the final keys saved are controlled by the"
            " TrapicheWorkflowParams."
        ),
    )
    bool_opt = argparse.BooleanOptionalAction
    p.add_argument(
        "--run-text",
        dest="run_text",
        action=bool_opt,
        default=None,
        help="Enable or disable text prediction step (env: TRAPICHE_RUN_TEXT)",
    )
    p.add_argument(
        "--run-vectorise",
        dest="run_vectorise",
        action=bool_opt,
        default=None,
        help="Enable or disable vectorisation step (env: TRAPICHE_RUN_VECTORISE)",
    )
    p.add_argument(
        "--run-taxonomy",
        dest="run_taxonomy",
        action=bool_opt,
        default=None,
        help="Enable or disable taxonomy prediction step (env: TRAPICHE_RUN_TAXONOMY)",
    )

    # Text params
    p.add_argument(
        "--sample-study-text-heuristic",
        dest="sample_study_text_heuristic",
        action=bool_opt,
        default=None,
        help=(
            "When set, if both project_description_text and sample_description_text are provided, "
            "run predictions on both and take union the labels; "
        ),
    )

    p.set_defaults(disable_minimal_result=False)

    # Model configuration: flags and/or config file
    p.add_argument(
        "--config",
        dest="config_file",
        default=None,
        help=(
            "Path to a dotenv-style config file setting TRAPICHE_* variables "
            "(e.g. TRAPICHE_TEXT_HF_MODEL, TRAPICHE_TAXONOMY_LOCAL_MODEL_DIR). "
            "Loaded before any other flags/env vars are read; explicit CLI "
            "flags below take precedence over values from this file."
        ),
    )
    model_group = p.add_argument_group(
        "model configuration",
        "Select model repo/version, or point to a local directory to bypass "
        "Hugging Face Hub downloads (offline use). Each can also be set via "
        "the corresponding TRAPICHE_* environment variable or --config file.",
    )
    model_group.add_argument(
        "--text-hf-model",
        dest="text_hf_model",
        default=None,
        help="HF repo id for the text classifier (env: TRAPICHE_TEXT_HF_MODEL)",
    )
    model_group.add_argument(
        "--text-model-version",
        dest="text_model_version",
        default=None,
        help="Version/tag of the text classifier (env: TRAPICHE_TEXT_MODEL_VERSION)",
    )
    model_group.add_argument(
        "--text-local-model-dir",
        dest="text_local_model_dir",
        default=None,
        help=(
            "Local directory (mirroring the HF repo layout) to load the text "
            "classifier from instead of Hugging Face Hub "
            "(env: TRAPICHE_TEXT_LOCAL_MODEL_DIR)"
        ),
    )
    model_group.add_argument(
        "--vector-hf-model",
        dest="vector_hf_model",
        default=None,
        help="HF repo id for the community2vec vectorizer (env: TRAPICHE_VECTOR_HF_MODEL)",
    )
    model_group.add_argument(
        "--vector-model-version",
        dest="vector_model_version",
        default=None,
        help="Version/tag of the vectorizer (env: TRAPICHE_VECTOR_MODEL_VERSION)",
    )
    model_group.add_argument(
        "--vector-local-model-dir",
        dest="vector_local_model_dir",
        default=None,
        help=(
            "Local directory (mirroring the HF repo layout) to load the "
            "vectorizer from instead of Hugging Face Hub "
            "(env: TRAPICHE_VECTOR_LOCAL_MODEL_DIR)"
        ),
    )
    model_group.add_argument(
        "--taxonomy-hf-model",
        dest="taxonomy_hf_model",
        default=None,
        help="HF repo id for the taxonomy classifier (env: TRAPICHE_TAXONOMY_HF_MODEL)",
    )
    model_group.add_argument(
        "--taxonomy-model-version",
        dest="taxonomy_model_version",
        default=None,
        help="Version/tag of the taxonomy classifier (env: TRAPICHE_TAXONOMY_MODEL_VERSION)",
    )
    model_group.add_argument(
        "--taxonomy-local-model-dir",
        dest="taxonomy_local_model_dir",
        default=None,
        help=(
            "Local directory (mirroring the HF repo layout) to load the "
            "taxonomy classifier from instead of Hugging Face Hub "
            "(env: TRAPICHE_TAXONOMY_LOCAL_MODEL_DIR)"
        ),
    )

    # Logging option: default to trapiche.log when running via the CLI
    p.add_argument(
        "--log-file",
        dest="log_file",
        default="trapiche.log",
        help=("Path to log file (defaults to 'trapiche.log')."),
    )

    # add vervose option to set logger level
    p.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
        dest="log_level",
        help="Enable verbose logging output (DEBUG level).",
    )

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the trapiche CLI.

    Args:
        argv: Optional list of command-line arguments.

    Returns:
        int: Process exit code (0 on success).
    """
    args = parse_args(argv)

    # Load config file (if any) before anything else reads TRAPICHE_* env
    # vars, then apply explicit CLI flags on top (flags take precedence).
    if args.config_file:
        load_config_file(args.config_file)
    for dest, env_var in _MODEL_ENV_VARS.items():
        value = getattr(args, dest, None)
        if value is not None:
            os.environ[env_var] = value

    logfile = args.log_file
    setup_logging(logfile=logfile, level=args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("trapiche CLI invoked | command_line_arguments='%s'", " ".join(sys.argv))

    logger.info("Parsed arguments: %s", args)

    inpath = None
    if args.input and args.input != "-":
        inpath = Path(args.input)

    outpath = None
    if args.output:
        outpath = Path(args.output)

    if not getattr(args, "disable_minimal_result", False):
        # minimal result enabled -> use compact output_keys from config
        output_keys = TrapicheWorkflowParams().output_keys
    else:
        # minimal result explicitly disabled
        output_keys = None

    # Load defaults from env via Pydantic, then override with CLI if provided
    base_params = TrapicheWorkflowParams()
    update_fields: dict[str, Any] = {}
    if args.run_text is not None:
        update_fields["run_text"] = bool(args.run_text)
    if args.run_vectorise is not None:
        update_fields["run_vectorise"] = bool(args.run_vectorise)
    if args.run_taxonomy is not None:
        update_fields["run_taxonomy"] = bool(args.run_taxonomy)
    if args.sample_study_text_heuristic is not None:
        update_fields["sample_study_text_heuristic"] = bool(args.sample_study_text_heuristic)
    # Output keys are controlled by the CLI flag above
    update_fields["output_keys"] = output_keys

    workflow_params = base_params.model_copy(update=update_fields)

    # Model params are constructed after the env vars above are set, so
    # CLI flags / --config values are reflected here.
    text_params = TextToBiomeParams()
    vectorise_params = TaxonomyToVectorParams()
    taxonomy_params = TaxonomyToBiomeParams()

    # read input
    samples = list(read_ndjson(inpath))
    if not samples:
        # nothing to do, write empty output
        # If no output path specified but an input file was used, create a
        # default output filename based on the input basename.
        if outpath is None and inpath is not None:
            # strip all suffixes from input name (e.g. .tsv.gz -> base)
            base_path = inpath
            while base_path.suffix:
                base_path = base_path.with_suffix("")
            outpath = inpath.parent / f"{base_path.name}_trapiche_results.ndjson"

        write_ndjson([], outpath)
        return 0

    runner = TrapicheWorkflowFromSequence(
        workflow_params=workflow_params,
        text_params=text_params,
        vectorise_params=vectorise_params,
        taxonomy_params=taxonomy_params,
    )
    processed = runner.run(samples)

    # If no output path specified but an input file was used, generate
    # a default filename: <input_basename>_trapiche_results.ndjson
    if outpath is None and inpath is not None:
        base_path = inpath
        while base_path.suffix:
            base_path = base_path.with_suffix("")
        outpath = inpath.parent / f"{base_path.name}_trapiche_results.ndjson"

    # processed is a sequence of dicts

    logger.info(f"Writing results to | output_path={outpath if outpath else 'stdout'}")
    write_ndjson(processed, outpath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
