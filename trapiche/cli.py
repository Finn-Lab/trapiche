
"""Command line interface for Trapiche predictions.

Exposes a single entry point ``main`` (fastcore call_parse) plus helper
utilities for validating input files and reading study descriptions.
"""

from __future__ import annotations

from .utils import load_biome_herarchy_dict

__all__ = [
    "print_help_extended",
    "check_input_file_format",
    "find_taxonomy_files",
    "read_description_file",
    "main",
]

import logging
import os
import sys
import argparse
from typing import Dict, List, Iterable

import numpy as np
import pandas as pd

from . import model_registry
from .deep_pred import predict_runs
from .trapiche_text import TextClassifier

def print_help_extended() -> None:
    """Print extended help information for the Trapiche CLI tool."""
    help_text = """
Trapiche CLI Tool Help:

Usage:
    python trapiche.py sample_dir <path> [options]

Options:
    --input_file <path>          Required. Specify the path to a TSV file (without a header). Each row in the file corresponds to a sample. 
                                  Column 1 should contain the directory path where the taxonomy files are located. Expected file extensions for taxonomy predictions are *.mseq.txt and *.diamond.tsv[.gz].
                                  and Column 2 should contain the path to a text file with the study description. Expected file extension for text files is *.txt.
    --taxonomy_prediction <bool> Optional. Perform taxonomy-based prediction using files in the provided directories.
                                  Default is True. Set to False if you do not want to perform this analysis.
                                  Expected file extensions for taxonomy predictions are *.mseq.txt and *.diamond.tsv[.gz].
    --text_prediction <bool>     Optional. Perform text-based prediction using study description files in the directories.
                                  Default is True. Set to False if you do not want to perform this analysis.
                                  Expected file extension for text files is *.STUDY_DESCRIPTION.txt.
    --save_comm2vec <bool>       Optional. Save the embedding of the microbial community if set to True.
                                  Default is False.
    --output_dir <path>          Optional. Specify the directory where results should be saved. If not provided,
                                  results will be saved in a directory named <current_directory>_TRAPICHE.
    --help_extended              Show this help message and exit.

Description:
    The Trapiche CLI tool processes directories containing taxonomy annotation and text annotation files,
    performs predictions based on these files, logs the activities, and outputs the results. The tool is designed
    to handle either a single directory or multiple directories listed in a text file. The directories should contain
    files relevant to the analysis types enabled via options.

Examples:
    trapiche --input_file test/files/input_file.tsv
    trapiche --input_file test/files/input_file.tsv --save_comm2vec True --output_dir /path/to/output
    trapiche --input_file test/files/input_file.tsv --text_prediction False --taxonomy_prediction True
    """
    print(help_text)

def check_input_file_format(file_path: str) -> pd.DataFrame | bool:
    """Validate the two-column TSV mapping taxonomy directories to text files.

    Returns the dataframe with proper column names or ``False`` when invalid.
    """
    try:
        df = pd.read_csv(file_path, sep='\t', header=None)
        if df.shape[1] != 2:
            print("Error: The input file must have exactly two columns.")
            return False
        for directory in df[0]:
            if not os.path.isdir(directory):
                print(f"Error: {directory} is not a valid directory.")
                return False
        for file in df[1]:
            if not os.path.isfile(file) or not file.endswith('.txt'):
                print(f"Error: {file} is not a valid text file.")
                return False
        df.columns = ['taxonomy_files_dir','study_description_file']
        return df
    except Exception as e:
        print(f"Error: {e}")
        return False


def find_taxonomy_files(directory: str) -> List[str]:
    """Recursively collect supported taxonomy annotation files for a directory."""
    taxo_files: List[str] = []
    for root, _dirs, files in os.walk(directory):
        for file in files:
            full = os.path.join(root, file)
            # Accept krona-style marker gene summaries
            if file.endswith(".mseq.txt"):
                taxo_files.append(full)
                continue
            # Accept diamond outputs: common patterns include *_diamond.tsv or *_diamond.tsv.gz
            if "diamond" in file and (file.endswith(".tsv") or file.endswith(".tsv.gz")):
                taxo_files.append(full)
                continue
    if not taxo_files:
        logging.getLogger(__name__).warning(
            "No taxonomy files detected under %s (looked for *.mseq.txt, *diamond*.tsv[.gz])",
            directory,
        )
    return sorted(taxo_files)

def read_description_file(file_path: str) -> str:
    """Read a UTF-8 (fallback ISO-8859-1) encoded description file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            study_description = f.read().strip()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='ISO-8859-1') as f:
            study_description = f.read().strip()
    return study_description


def main(argv: Iterable[str] | None = None) -> None:
    """Run Trapiche prediction workflow or download models (argparse variant).
    This mirrors the previous fastcore call_parse interface while adopting the argparse style used by the text classifier CLI for consistency.
    """

    # Logging setup: expose log level variable, override with env
    log_level = os.environ.get("TRAPICHE_LOG_LEVEL", "INFO").upper()
    parser = argparse.ArgumentParser(
        description="Trapiche biome prediction (taxonomy + text).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_file", required=True, help="Two-column TSV: taxonomy dir, study description file (no header)")
    parser.add_argument("--output_dir", default=None, help="Destination directory for outputs (auto-created)")
    parser.add_argument("--no_taxonomy_prediction", action="store_true", help="Skip taxonomy-based prediction")
    parser.add_argument("--no_text_prediction", action="store_true", help="Skip text-based prediction")
    parser.add_argument("--save_comm2vec", action="store_true", help="Save community embedding matrix (when taxonomy prediction enabled)")
    parser.add_argument("--log_file", default="trapiche.log", help="Log file name inside output directory")
    parser.add_argument("--download_models", action="store_true", help="Download / validate models and exit")
    parser.add_argument("--help_extended", action="store_true", help="Show extended help and exit")
    parser.add_argument("--log_level", default=None, help="Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")

    args = parser.parse_args(list(argv) if argv is not None else None)

    # CLI arg overrides env
    if args.log_level:
        log_level = args.log_level.upper()

    # Output dir setup
    input_file = args.input_file
    output_dir = args.output_dir
    if output_dir in (None, "None", ""):
        output_dir = os.path.join(
            os.getcwd(), os.path.basename(input_file) + "_TRAPICHE_RESULTS"
        )
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(input_file) + "_TRAPICHE_RESULTS"

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        filename=f"{output_dir}/{args.log_file}",
        filemode="a",
    )
    logger = logging.getLogger(__name__)
    logger.info(f"trapiche CLI startup log_level={log_level} output_dir={output_dir}")

    if args.help_extended:
        print_help_extended()
        logger.info("help_extended requested; exiting")
        sys.exit(0)

    if args.download_models:
        logger.info("Downloading/validating models into cache ...")
        try:
            model_registry.download_models()
            logger.info(f"Models ready under cache_root={model_registry.cache_root()}")
            print(f"Models ready under {model_registry.cache_root()}")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Model download failed error={e}")
            print(f"Model download failed: {e}")
            sys.exit(1)

    input_df = check_input_file_format(input_file)
    if not isinstance(input_df, pd.DataFrame):
        logger.error("Input file format is incorrect. Please provide a valid file.")
        print("Input file format is incorrect. Please provide a valid file.")
        sys.exit(1)

    logger.info(f"input_file={input_file} n_samples={input_df.shape[0]}")
    logger.debug(f"input_df columns={input_df.columns.tolist()} head={input_df.head().to_dict()}")

    # Text-based prediction (optional)
    biome_herarchy_dct = load_biome_herarchy_dict()
    if not args.no_text_prediction:
        logger.info("Starting text-based prediction")
        text_classifier = TextClassifier(model_path="SantiagoSanchezF/trapiche-biome-classifier")
        study_description: Dict[str, str] = {}
        for study_description_file in input_df['study_description_file'].unique():
            logger.info(f"Reading text file file={study_description_file}")
            study_description[study_description_file] = read_description_file(study_description_file)
        logger.info(f"Making text-based predictions n_files={len(study_description)}")
        text_predictions = text_classifier.predict(list(study_description.values()))
        text_predictions_amended = [
            [biome_herarchy_dct.get(biome, biome) for biome in study_text_prediction]
            for study_text_prediction in text_predictions
        ]
        text_results = dict(zip(study_description.keys(), text_predictions_amended))
        logger.debug(f"text_results={text_results}")
    else:
        logger.info("Not performing text-based prediction")
        text_results = {}
    input_df['sample_text_constrains'] = input_df['study_description_file'].map(lambda x: text_results.get(x, []))

    # Taxonomy based predictions -------------------------------------------------
    list_of_taxonomy_files: List[List[str] | None] = []
    if not args.no_taxonomy_prediction:
        logger.info("Finding files for taxonomy-based prediction")
        for directory in input_df['taxonomy_files_dir']:
            logger.info(f"Scanning directory dir={directory}")
            taxonomy_files = find_taxonomy_files(directory)
            logger.info(f"Found n_files={len(taxonomy_files)} in dir={directory}")
            logger.debug(f"taxonomy_files={taxonomy_files}")
            list_of_taxonomy_files.append(taxonomy_files)
        logger.info("Predicting based on taxonomy data...")
        df, c2v_mat = predict_runs(
            list_of_taxonomy_files,
            return_full_preds=True,
            constrain=input_df['sample_text_constrains'].tolist(),
        )
        result_df = pd.concat([input_df, df], axis=1)
        logger.debug(f"taxonomy prediction result_df columns={result_df.columns.tolist()} head={result_df.head().to_dict()}")
        if c2v_mat is not None:
            logger.info(f"Community vector matrix shape={getattr(c2v_mat, 'shape', None)}")
    else:
        list_of_taxonomy_files = [None] * input_df.shape[0]
        result_df = input_df.copy()
        for col in ("lineage_pred", "refined_prediction", "unbiased_taxo_prediction"):
            result_df[col] = [None] * result_df.shape[0]
        logger.info("Not performing taxonomy-based prediction")

    result_df['taxonomy_files'] = list_of_taxonomy_files
    # Stringify list columns for TSV export
    result_df['sample_text_constrains'] = ["|".join(dirs) for dirs in result_df['sample_text_constrains']]
    result_df['taxonomy_files'] = ["|".join(files) if files else "" for files in result_df['taxonomy_files']]

    # Save outputs --------------------------------------------------------------
    result_df.to_csv(f"{output_dir}/{basename}.results.tsv", sep="\t", index=False)
    logger.info(f"Results saved to file={output_dir}/{basename}.results.tsv shape={result_df.shape}")

    if args.save_comm2vec and not args.no_taxonomy_prediction:
        np.save(f"{output_dir}/{basename}.c2v.npy", c2v_mat)  # type: ignore[name-defined]
        logger.info(f"Community vector matrix saved to {output_dir}/{basename}.c2v.npy")

    result_df[["taxonomy_files_dir","study_description_file", "sample_text_constrains", "unbiased_taxo_prediction", "lineage_pred" ,"refined_prediction"]].to_csv(
        f"{output_dir}/{basename}.minimal_results.tsv", sep="\t", index=False
    )
    logger.info(f"Minimal results saved to file={output_dir}/{basename}.minimal_results.tsv")
    logger.info("Processing complete.")
    print("Processing complete.")

if __name__ == "__main__":  # pragma: no cover
    main()
