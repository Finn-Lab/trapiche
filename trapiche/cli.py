from __future__ import annotations
import argparse
import sys
import json
from pathlib import Path
from typing import Iterable, Dict, Any
import gzip

from .api import TrapicheWorkflowFromSequence
from .config import TrapicheWorkflowParams


def read_ndjson(path: Path | None) -> Iterable[Dict[str, Any]]:
    """Yield objects from NDJSON file or stdin.

    If path is None, read from stdin. Supports .gz compressed files.
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
    """Write records as NDJSON to path or stdout."""
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
    p.add_argument("-o", "--output", help="Output NDJSON file path (defaults to stdout). Use .gz to compress")

    # workflow toggles mirroring TrapicheWorkflowParams
    p.add_argument("--no-text", dest="run_text", action="store_false", help="Do not run text prediction step")
    p.add_argument("--keep-text-results", dest="keep_text_results", action="store_true", help="Keep text intermediate results in output")
    p.add_argument("--no-vectorise", dest="run_vectorise", action="store_false", help="Do not run vectorisation step")
    p.add_argument("--keep-vectorise-results", dest="keep_vectorise_results", action="store_true", help="Keep vectorisation intermediate results in output")
    p.add_argument("--no-taxonomy", dest="run_taxonomy", action="store_false", help="Do not run taxonomy prediction step")
    p.add_argument("--keep-taxonomy-results", dest="keep_taxonomy_results", action="store_true", help="Keep taxonomy intermediate results in output")

    p.add_argument(
        "--minimal-result",
        dest="minimal_result",
        action="store_true",
        help=("When set, keep the minimal output."
              "When not set, final keys are controlled by the --keep-*-results flags."),
    )

    # defaults mirror dataclass defaults from TrapicheWorkflowParams
    defaults = TrapicheWorkflowParams()
    p.set_defaults(
        run_text=defaults.run_text,
        keep_text_results=defaults.keep_text_results,
        run_vectorise=defaults.run_vectorise,
        keep_vectorise_results=defaults.keep_vectorise_results,
        run_taxonomy=defaults.run_taxonomy,
        keep_taxonomy_results=defaults.keep_taxonomy_results,
        minimal_result=True,
    )

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    inpath = None
    if args.input and args.input != "-":
        inpath = Path(args.input)

    outpath = None
    if args.output:
        outpath = Path(args.output)

    # build params dataclass
    # Determine output_keys according to --minimal-result
    if args.minimal_result:
        output_keys = TrapicheWorkflowParams().output_keys
    else:
        output_keys = None

    params = TrapicheWorkflowParams(
        run_text=bool(args.run_text),
        keep_text_results=bool(args.keep_text_results),
        run_vectorise=bool(args.run_vectorise),
        keep_vectorise_results=bool(args.keep_vectorise_results),
        run_taxonomy=bool(args.run_taxonomy),
        keep_taxonomy_results=bool(args.keep_taxonomy_results),
        output_keys=output_keys,
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

    runner = TrapicheWorkflowFromSequence(params=params)
    processed = runner.run(samples)

    # If no output path specified but an input file was used, generate
    # a default filename: <input_basename>_trapiche_results.ndjson
    if outpath is None and inpath is not None:
        base_path = inpath
        while base_path.suffix:
            base_path = base_path.with_suffix("")
        outpath = inpath.parent / f"{base_path.name}_trapiche_results.ndjson"

    # processed is a sequence of dicts
    write_ndjson(processed, outpath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
