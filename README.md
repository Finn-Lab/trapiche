# Trapiche
> A Biome Classification Tool for Metagenomic Datasets

Trapiche is a hybrid method designed to classify metagenomic samples into specific biomes according to the [GOLD Ecosystem Classification](https://doi.org/10.1093/nar/gkaa983). It can utilize either text or taxonomy information independently to classify samples. However, integrating both text and taxonomy information yields the most accurate results.

## Installation

To ensure a smooth installation process and avoid conflicts with other packages, we highly recommend installing Trapiche in a dedicated virtual environment. Follow these steps to install Trapiche:

```sh
pip install .
```

## Basic usage

```py

import glob

with open(file_with_text_description_of_project) as h: # 
    prj_desc = h.read()


from trapiche.llm_layer import lineages_from_text
from trapiche.deep_pred import predict_runs as deep_predict_runs

text_prediction = lineages_from_text(prj_desc)

taxonomy_files = glob.glob(f"{directory_with_taxonomy_annotations_files}/*mseq.txt")

result_dataframe = deep_predict_runs([[f] for f in taxonomy_files],return_full_preds=True,constrain=[prj_desc for _ in taxonomy_files])
```
