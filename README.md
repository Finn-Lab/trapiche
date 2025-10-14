# Trapiche

Lightweight toolkit to predict environmental biomes from sample descriptions
and taxonomic annotations. This repository provides small, focused building
blocks and a simple command line interface to run them.

Features
- TextToBiome: multi-label text classifier using a BERT model (HuggingFace).
- Community2vec: vectorise taxonomy annotation files into community vectors.
- TaxonomyToBiome: deep predictor converting community vectors into biome
    lineages; can be constrained by text predictions.

CLI

Install locally and run the `trapiche` console script (configured in
`settings.ini`). The CLI reads NDJSON (newline-delimited JSON) and writes
NDJSON results. Each input line should be a JSON object with at least:

- `project_description_file_path` (optional): path to a text file
- `taxonomy_files_paths` (optional): array of paths to taxonomy TSV files

Example input line:

```json
{"project_description_file_path": "test/files/text_files/PRJEB42572_project_description.txt", "taxonomy_files_paths": ["test/files/taxonomy_files/ERZ19590789/ERZ19590789_FASTA_diamond.tsv"]}
```

Run full workflow (stdin -> stdout):

```bash
cat samples.ndjson | trapiche --run workflow > results.ndjson
```

Run only the text step:

```bash
trapiche --input samples.ndjson --run text > text_results.ndjson
```

Output

Each output JSON object is the input object augmented with optional keys:

- `text_predictions`: list of predicted biome labels from text
- `community_vector`: numeric vector (list) representing the sample community
- `taxonomy_prediction`: model output (dict/row) from TaxonomyToBiome

Notes for developers
- API wrappers live in `trapiche/api.py`.
- The orchestrator is `trapiche/workflow.py` and the CLI is
    `trapiche/cli.py`.
