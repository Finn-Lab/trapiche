"""Command-line interface for running the Trapiche workflow.

Reads NDJSON input (file or stdin), executes selected steps, and writes
NDJSON output (file or stdout). Supports gzip input/output.
"""
from __future__ import annotations
import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Iterable, Dict, Any
import gzip

from .api import TrapicheWorkflowFromSequence
from .config import TrapicheWorkflowParams, TaxonomyToVectorParams, setup_logging


def read_ndjson(path: Path | None) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from NDJSON input.

    Args:
        path: Input file path or None to read from stdin. .gz supported.

    Yields:
        dict: One object per line.
    """
    fh = None
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
            fh = gzip.open(path, "rt", encoding="utf-8")
        else:
            fh = open(path, "r", encoding="utf-8")

        with fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON encountered: {e}")
    except FileNotFoundError:
        raise SystemExit(f"Input file not found: {path}")


def write_ndjson(records: Iterable[Dict[str, Any]], path: Path | None) -> None:
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
    p.add_argument("input", nargs="?", help="Input NDJSON file path (use - for stdin). Supports .gz")
    p.add_argument("-o", "--output", help="Output NDJSON file path (defaults to <INPUT_BASENAME>_trapiche_results.ndjson). Use .gz to compress")
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
    # workflow toggles mirroring TrapicheWorkflowParams
    p.add_argument("--no-text", dest="run_text", action="store_false", help="Do not run text prediction step")
    p.add_argument("--no-vectorise", dest="run_vectorise", action="store_false", help="Do not run vectorisation step")
    p.add_argument("--no-taxonomy", dest="run_taxonomy", action="store_false", help="Do not run taxonomy prediction step")
    # Text params
    p.add_argument(
        "--sample-study-text-heuristic",
        dest="sample_study_text_heuristic",
        action="store_true",
        help=(
            "When set, if both project_description_text and sample_description_text are provided, "
            "run predictions on both and take union the labels; "
        ),
    )



    # defaults mirror dataclass defaults from TrapicheWorkflowParams
    defaults = TrapicheWorkflowParams()
    p.set_defaults(
        run_text=defaults.run_text,
        run_vectorise=defaults.run_vectorise,
        run_taxonomy=defaults.run_taxonomy,
        disable_minimal_result=False,
        sample_study_text_heuristic=defaults.sample_study_text_heuristic,
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

    logfile = args.log_file
    setup_logging(logfile=logfile, level=args.log_level)
    logger = logging.getLogger(__name__)

    logger.info(
        "trapiche CLI invoked | command_line_arguments='%s'", " ".join(sys.argv)
    )
   
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

    params = TrapicheWorkflowParams(
        run_text=bool(args.run_text),
        run_vectorise=bool(args.run_vectorise),
        run_taxonomy=bool(args.run_taxonomy),
        output_keys=output_keys,
        sample_study_text_heuristic=bool(args.sample_study_text_heuristic),
    )


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

    # Determine taxonomy vectorization model params from config defaults
    _t2v = TaxonomyToVectorParams()
    runner = TrapicheWorkflowFromSequence(workflow_params=params)
    processed = runner.run(samples, model_name=_t2v.hf_model, model_version=_t2v.model_version)

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
