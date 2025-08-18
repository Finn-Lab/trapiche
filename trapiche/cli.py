
__all__ = ['print_help_extended', 'check_input_file_format', 'find_taxonomy_files', 'read_description_file', 'main']

# %% ../nbs/01.00.06_CLI.ipynb 2
import sys
import os
import logging
import pandas as pd
import numpy as np
from fastcore.script import *
from .deep_pred import predict_runs
from .goldOntologyAmendments import biome_herarchy_dct
from .trapiche_text import TextClassifier

# %% ../nbs/01.00.06_CLI.ipynb 3
def print_help_extended():
    """
    Print help information for the Trapiche CLI tool.
    """
    help_text = """
Trapiche CLI Tool Help:

Usage:
    python trapiche.py sample_dir <path> [options]

Options:
    --input_file <path>          Required. Specify the path to a TSV file (without a header). Each row in the file corresponds to a sample. 
                                  Column 1 should contain the directory path where the taxonomy files are located. Expected file extensions for taxonomy predictions are *.mseq.txt and *.diamond.tsv.gz.
                                  and Column 2 should contain the path to a text file with the study description. Expected file extension for text files is *.txt.
    --taxonomy_prediction <bool> Optional. Perform taxonomy-based prediction using files in the provided directories.
                                  Default is True. Set to False if you do not want to perform this analysis.
                                  Expected file extensions for taxonomy predictions are *.mseq.txt and *.diamond.tsv.gz.
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

# %% ../nbs/01.00.06_CLI.ipynb 4
def check_input_file_format(file_path):
    try:
        # Read the TSV file
        df = pd.read_csv(file_path, sep='\t', header=None)

        # Check if the file has exactly two columns
        if df.shape[1] != 2:
            print("Error: The input file must have exactly two columns.")
            return False

        # Check if all paths in column 1 are directories
        for directory in df[0]:
            if not os.path.isdir(directory):
                print(f"Error: {directory} is not a valid directory.")
                return False

        # Check if all paths in column 2 are text files
        for file in df[1]:
            if not os.path.isfile(file) or not file.endswith('.txt'):
                print(f"Error: {file} is not a valid text file.")
                return False

        df.columns = ['taxonomy_files_dir','study_description_file']
        return df
    except Exception as e:
        print(f"Error: {e}")
        return False


# %% ../nbs/01.00.06_CLI.ipynb 5
def find_taxonomy_files(directory):
    taxo_files_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".mseq.txt") or file.endswith(".diamond.tsv.gz"):
                taxo_files_list.append(
                    os.path.join(root, file)
                )
    return taxo_files_list

# %% ../nbs/01.00.06_CLI.ipynb 6
def read_description_file(file_path):
    " function to read description file, handling different encodings"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            study_description = f.read().strip()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='ISO-8859-1') as f:
            study_description = f.read().strip()
    return study_description

# %% ../nbs/01.00.06_CLI.ipynb 7
@call_parse
def main(
    help_extended: bool=False,  # Show extended help message and exit.
    input_file: str = 'None',  # MANDATORY TO RUN PREDICTION. Path to a tsv file (no header) where row is for a sample and column 1 is a directory where to find taxonomy files and column 2 is a text file with the study description.
    output_dir: str = 'None',  # Output directory in which results will be saved. If False, the current directory will be used.
    no_taxonomy_prediction: bool = False,  # Use taxonomy annotation files in provided directories for prediction. If False, provide dummy path.
    no_text_prediction: bool = False,  # Perform prediction based on text annotation. If False, provide dummy path.
    save_comm2vec: bool = False,  # Write the embedding of the microbial community.
    log_file: str = "trapiche.log",  # Log file name.
):
    """
    Main function of CLI
    """

    if help_extended:
        print_help_extended()
        sys.exit(0)
    
    if input_file == 'None':
        print("input_file is needed for prediction")
        print_help_extended()
        sys.exit(0)
        
    # Set output directory
    if output_dir == 'None':
        output_dir = os.path.join(
            os.getcwd(), os.path.basename(input_file) + "_TRAPICHE_RESULTS"
        )
        os.makedirs(output_dir, exist_ok=True)

    basename = os.path.basename(input_file) + "_TRAPICHE_RESULTS"
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filename=f"{output_dir}/{log_file}",
        filemode="a",
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting Trapiche processing...")
    logger.info(f"Results and logfile will be saved in {output_dir}")
    input_df = check_input_file_format(input_file)
    
    # Input sanity check
    if type(input_df)!=pd.core.frame.DataFrame:
        print("Input file format is incorrect. Please provide a valid file.")
        logger.info("Input file format is incorrect. Please provide a valid file.")
        sys.exit(1)


    # Text-based prediction
    if not no_text_prediction:
        logger.info(f"Starting text-based prediction")
        text_classifier = TextClassifier(model_path="SantiagoSanchezF/trapiche-biome-classifier")
        study_description = {}
        for study_description_file in input_df['study_description_file'].unique():
            logger.info(f"Reading text file {study_description_file}")
            study_description[study_description_file] = read_description_file(study_description_file)
        logger.info(f"Making text-based predictions")
        text_predictions = text_classifier.predict(list(study_description.values()))
        # Transform GOLD prediction to GOLDamended
        text_predictions_amended = [
            [biome_herarchy_dct.get(biome, biome) for biome in study_text_prediction]
            for study_text_prediction in text_predictions
        ]
        text_results = dict(zip(study_description.keys(),text_predictions_amended))
    else:
        logger.info(f"Not performing text-based prediction")
    input_df['sample_text_constrains'] = input_df['study_description_file'].map(lambda x: text_results.get(x,[]))

    # taxonomy predictions
    list_of_taxonomy_files = []
    if not no_taxonomy_prediction:
        logger.info(f"Finding files for taxonomy-based prediction")
        for directory in input_df['taxonomy_files_dir']:
            taxonomy_files = []
            logger.info(f"Scanning directory: {directory}")
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith(".mseq.txt") or file.endswith("diamond.tsv.gz"):
                        taxonomy_files.append(
                            os.path.join(root, file)
                        )
            logger.info(
                f"Found {len(taxonomy_files)} taxonomy files in directory {directory}"
            )
            list_of_taxonomy_files.append(taxonomy_files)
        
        logger.info("Predicting based on taxonomy data...")
        df, c2v_mat = predict_runs(
            list_of_taxonomy_files,
            return_full_preds=True,
            constrain=input_df['sample_text_constrains'].tolist(),
        )
        result_df = pd.concat([input_df, df], axis=1)
    else:
        list_of_taxonomy_files = [None for _ in range(input_df.shape[0])]
        result_df = input_df.copy()
        result_df["lineage_pred"] = [None for _ in range(result_df.shape[0])]
        result_df["refined_prediction"] = [None for _ in range(result_df.shape[0])]
        result_df["unbiased_taxo_prediction"] = [None for _ in range(result_df.shape[0])]
        logger.info(f"Not performing taxonomy-based prediction")
    
    result_df['taxonomy_files'] = list_of_taxonomy_files

    # Add constrains to final dataframe
    result_df['sample_text_constrains'] = ["|".join(dirs) for dirs in result_df['sample_text_constrains']]
    result_df['taxonomy_files'] = ["|".join(files) for files in result_df['taxonomy_files']]

    # Save results
    result_df.to_csv(f"{output_dir}/{basename}.results.tsv", sep="\t", index=None)
    logger.info(f"Results saved to {output_dir}/{basename}.results.tsv")

    if save_comm2vec:
        np.save(f"{output_dir}/{basename}.c2v.npy", c2v_mat)
        logger.info(f"Community vector matrix saved to {output_dir}/{basename}.c2v.npy")

    # Minimal results output
    result_df[["taxonomy_files_dir","study_description_file", "sample_text_constrains", "unbiased_taxo_prediction", "lineage_pred" ,"refined_prediction"]].to_csv(
        f"{output_dir}/{basename}.minimal_results.tsv", sep="\t", index=None
    )

    logger.info("Processing complete.")
    print("Processing complete.")
