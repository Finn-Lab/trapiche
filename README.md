Trapiche — Joint biome classification from text and taxonomy
===========================================================

Trapiche is an open-source tool for biome classification in metagenomic studies. It combines two complementary signals:

- Text-based context: an adapted Large Language Model (LLM) predicts candidate biomes from project/sample descriptions.
- Taxonomy-based signal: a community2vec embedding of taxonomic profiles is fed to a feed‑forward model for deep biome lineage prediction.

By integrating both views, Trapiche improves accuracy and robustness in biome classification workflows.


## Install

Requirements
- Python 3.10+
- Linux/macOS recommended (CPU or CUDA GPU)

From source
1) Clone this repository
2) Install the package and dependencies

```
pip install -e .
```

Download models and resources

```
trapiche-download-models
```

This fetches the required community2vec resources, taxonomy graph, and other assets used at runtime.


## Quick start (CLI)

The CLI expects NDJSON (one JSON object per line). Each object represents one sample.

Required/optional keys per sample:

**Text predictions**
- project_description_text (optional): text describing the sample/project.
- project_description_file_path (optional): path to a text file with the description. If project_description_text is provided, this is ignored.

**Taxonomy predictions**
- taxonomy_files_paths (required for taxonomy predictions): list of file paths.
	- Accepted formats: .tsv, .tsv.gz (non-recursive).

Example input:

```
{"project_description_text":"Effect of different fertilization treatments on soil microbiome...", "taxonomy_files_paths":["test/files/taxonomy_files/ERZ34590789/ERZ34590789_FASTA_diamond.tsv.gz","test/files/taxonomy_files/ERZ34590789/ERZ34590789_FASTA_mseq.tsv"]}
{"project_description_file_path":"test/files/text_files/PRJEB42572_project_description.txt","taxonomy_files_paths":["test/files/taxonomy_files/ERZ19590789/ERZ19590789_FASTA_diamond.tsv.gz"]}
```

Run the workflow

```
# From file to default output path (<input>_trapiche_results.ndjson)
trapiche input.ndjson --minimal-result

# Or read from stdin and write to stdout
cat input.ndjson | trapiche -

# Keep intermediate results for inspection
trapiche input.ndjson --keep-text-results --keep-vectorise-results --keep-taxonomy-results

# Disable a step
trapiche input.ndjson --no-text  # no text-based constraints
```

Flags
- --no-text / --no-vectorise / --no-taxonomy
- --keep-text-results / --keep-vectorise-results / --keep-taxonomy-results
- --minimal-result (default: false). When set, output_keys defaults to a compact schema.


## Quick start (Python API)


End-to-end workflow over sample records

Uses a sequence of dicts (one dict is one sample) with the following required/optional keys per sample:

**Text predictions**
- project_description_text (optional): text describing the sample/project.
- project_description_file_path (optional): path to a text file with the description. If project_description_text is provided, this is ignored.

**Taxonomy predictions**
- taxonomy_files_paths (required for taxonomy predictions): list of file paths.
	- Accepted formats: .tsv, .tsv.gz (non-recursive).
    
```python
from trapiche.api import TrapicheWorkflowFromSequence
from trapiche.config import TrapicheWorkflowParams

samples = [
	{
		"project_description_file_path": "test/files/text_files/PRJEB42572_project_description.txt",
		"taxonomy_files_paths": [
			"test/files/taxonomy_files/ERZ19590789/ERZ19590789_FASTA_diamond.tsv.gz",
		],
	}
]

params = TrapicheWorkflowParams(  # defaults shown
	run_text=True, run_vectorise=True, run_taxonomy=True,
	keep_text_results=False, keep_vectorise_results=False, keep_taxonomy_results=False,
	# When output_keys is None, the keep_* flags decide what to include.
)

runner = TrapicheWorkflowFromSequence(params=params)
result = runner.run(samples)  # sequence of dicts augmented with predictions
runner.save("trapiche_results.ndjson")  # optional convenience save
```


Text prediction

```python
from trapiche.api import TextToBiome

ttb = TextToBiome()  # uses default model and device
preds = ttb.predict([
	"Soil metagenome from agricultural field in temperate climate.",
	"Gut microbiome samples from healthy adults.",
])
print(preds)  # list[list[str]]: predicted biome labels per input text

# Optionally save last predictions
ttb.save("text_preds.json")
```

Taxonomy → community vector → biome lineage

```python
from trapiche.api import Community2vec, TaxonomyToBiome

# Vectorise one or more samples from taxonomy annotation files
c2v = Community2vec()
vectors = c2v.transform([
	["test/files/taxonomy_files/ERZ19590789/ERZ19590789_FASTA_diamond.tsv"],
])

tax2b = TaxonomyToBiome()
df, vecs = tax2b.predict(community_vectors=vectors, return_full_preds=True)
print(df.head())  # pandas DataFrame with per-sample predictions

# Optional saves
c2v.save("community_vectors.npy")
tax2b.save("taxonomy_predictions.csv")
tax2b.save_vectors("taxonomy_vectors.npy")
```

## Input and output schema

Input record (API and CLI workflow)

```
{
	"taxonomy_files_paths": ["/path/to/sample1.tsv", "/path/to/sample1_b.tsv.gz"],
	"project_description_text": "Free text describing the sample.",
	# alternatively (if no inline text):
	# "project_description_file_path": "path/to/description.txt"
}
```

Output record (typical keys)

```
{
  ... original fields ...,
  "text_predictions": ["root:Host-Associated:Human", "root:Host-Associated:Animal"],                   # optional if run_text
  "community_vector": [0.12, -0.03, ...],                        # optional if keep_vectorise_results
  "lineage_prediction": "...",                                  # taxonomy-based prediction
  "lineage_prediction_probability": 0.93,
  "refined_prediction": "..."
}
```


## Data and models

Trapiche ships code only. Run `trapiche-download-models` once to download required resources (community2vec embeddings, taxonomy graph, and companion assets). Files are stored under an internal cache path managed by the library.


## Benchmarks (placeholder)

We will release a small benchmarking section here with datasets, metrics (accuracy, F1, calibration), and comparisons against text‑only and taxonomy‑only baselines.


## Paper (placeholder)

A preprint/manuscript describing Trapiche’s joint modelling approach and training data will be linked here.


## Contributing

Issues and pull requests are welcome. Please open an issue to discuss larger changes.


## License and citation

This project is open source. See the repository metadata for licensing terms.

If you use Trapiche in your work, please cite the Trapiche paper (coming soon). A BibTeX entry will be provided here.

