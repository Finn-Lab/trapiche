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

Model & Data Assets
===================

Large model/data artifacts are not shipped inside the Python package. They are fetched on demand and cached.

Cache location:
 - Default: ~/.cache/trapiche (override with TRAPICHE_CACHE env var)
 - Layout: <cache>/<model_name>/<version>/

Included models:
 - full_final_taxonomy.model.keras (FTP, ~30 GB)
 - mgnify_sample_vectors.h5 (FTP, ~200 MB)
 - trapiche-biome-classifier (HuggingFace Hub: SantiagoSanchezF/trapiche-biome-classifier)
 - en_core_sci_sm (SciSpacy biomedical model)

Download all models (may take significant time & disk space):

```
trapiche --download_models
```

Programmatic access:

```python
import trapiche
path = trapiche.get_model("mgnify_sample_vectors.h5", version="1.0", auto_download=True)
```

If a required model is missing during prediction, Trapiche will attempt to resolve it automatically (or raise with a helpful message).

Environment variables:
 - TRAPICHE_CACHE: set a custom cache directory.
