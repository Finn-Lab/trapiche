<p align="left">
  <img src="assets/logo.svg" width="350" alt="Trapiche logo">
</p>

Trapiche — Multi-source biome classification from text and taxonomy
===========================================================

Trapiche is an open-source tool for biome classification in metagenomic studies. It combines two complementary sources of information:

- Text-based: an adapted Large Language Model (LLM) predicts candidate biomes from project/sample descriptions.
- Taxonomy-based: a taxonomy_vectorization embedding of taxonomic profiles is fed to a feed‑forward model for deep biome lineage prediction.

By integrating both views, Trapiche improves accuracy and robustness in biome classification workflows.

## Install

Requirements
- Python 3.10+
- Linux/macOS recommended (CPU or CUDA GPU)

From source
1) Clone this repository
2) Install the package and dependencies

By default TensorFlow is optional. Choose the extra that matches your needs:

```bash
# Clone
git clone https://github.com/Finn-Lab/trapiche.git
cd trapiche

# Install without TensorFlow (default)
pip install .

# Install with CPU-only TensorFlow
pip install .[cpu]

# Install with GPU TensorFlow
pip install .[gpu]
```

## Quick start (CLI)

The CLI expects NDJSON (one JSON object per line). Each object represents one sample.

Required/optional keys per sample:

**Text predictions**
- project_description_text (optional): text describing the sample/project.
- project_description_file_path (optional): path to a text file with the description. If project_description_text is provided, this is ignored.
- sample_description_text (optional): additional text for the specific sample. Used when the sample-over-study heuristic is enabled.

**Taxonomy predictions**
- taxonomy_files_paths (required for taxonomy predictions): list of file paths.
	- Accepted formats: .tsv, .tsv.gz (non-recursive).

Example input:

```json
{"project_description_text":"Effect of different fertilization treatments on soil microbiome...", "taxonomy_files_paths":["test/files/taxonomy_files/ERZ34590789/ERZ34590789_FASTA_diamond.tsv.gz","test/files/taxonomy_files/ERZ34590789/ERZ34590789_FASTA_mseq.tsv"]}
{"project_description_file_path":"test/files/text_files/PRJEB42572_project_description.txt","taxonomy_files_paths":["test/files/taxonomy_files/ERZ19590789_FASTA_diamond.tsv.gz"]}
```

Run the workflow

```bash
# From file to default output path (<input>_trapiche_results.ndjson)
# By default the CLI writes a compact (minimal) result. To disable the
# minimal output and let the workflow params control which
# keys are saved, use the --disable-minimal-result flag.
trapiche input.ndjson

# To explicitly disable the minimal output and keep the full set controlled
# by TrapicheWorkflowParams:
trapiche input.ndjson --disable-minimal-result

# Or read from stdin and write to stdout
cat input.ndjson | trapiche -

# Disable a step
trapiche input.ndjson --no-run-text  # no text-based constraints

# Enable/disable the sample-over-study heuristic for text predictions
trapiche input.ndjson --sample-study-text-heuristic
trapiche input.ndjson --no-sample-study-text-heuristic
```

Flags
- `--run-text/--no-run-text`, `--run-vectorise/--no-run-vectorise`, `--run-taxonomy/--no-run-taxonomy`
- `--keep-text-results / --keep-vectorise-results / --keep-taxonomy-results`
- `--disable-minimal-result` (default: false). When set, the default minimal output is disabled and
	the final keys saved are controlled by `TrapicheWorkflowParams`. By default the CLI produces the compact/minimal
	output (no flag required).
- `--sample-study-text-heuristic` (or `--no-sample-study-text-heuristic`): when both project_description_text and sample_description_text are present, run text prediction on both and keep union of labels.

## Configuration via environment variables

Trapiche CLI and API use Pydantic Settings. You can override defaults with environment variables:

- `TRAPICHE_RUN_TEXT=true|false`
- `TRAPICHE_RUN_VECTORISE=true|false`
- `TRAPICHE_RUN_TAXONOMY=true|false`
- `TRAPICHE_SAMPLE_STUDY_TEXT_HEURISTIC=true|false`

Example:

```bash
export TRAPICHE_RUN_TEXT=false
export TRAPICHE_RUN_TAXONOMY=true
trapiche input.ndjson
```


## Quick start (Python API)


End-to-end workflow over sample records

Uses a sequence of dicts (one dict is one sample) with the following required/optional keys per sample:

**Text predictions**
- project_description_text (optional): text describing the sample/project.
- project_description_file_path (optional): path to a text file with the description. If project_description_text is provided, this is ignored.
- sample_description_text (optional): additional text describing the specific sample. Used only when the heuristic is enabled.

**Taxonomy predictions**
- taxonomy_files_paths (required for taxonomy predictions): list of file paths.
	- Accepted formats: .tsv, .tsv.gz (non-recursive).
    
```python
from trapiche.api import TrapicheWorkflowFromSequence
from trapiche.config import TrapicheWorkflowParams

samples = [
	{
		"project_description_text": "Home Microbiome Metagenomes. The project identifies patterns in microbial communities associated with different home and home occupant (human and pet) surfaces", 
		"sample_description_text": "Metagenome of microbial community: Bedroom Floor. House_04a-Bedroom_Floor_Day3. House_04a-Bedroom_Floor_Day3", 
		"taxonomy_files_paths": [
			"test/taxonomy_files/SRR1524511_MERGED_FASTQ_SSU_OTU.tsv", 
			"test/taxonomy_files/SRR1524511_MERGED_FASTQ_LSU_OTU.tsv"
			]
	}
]

workflow_params = TrapicheWorkflowParams(  # defaults shown
	run_text=True, run_vectorise=True, run_taxonomy=True,
	keep_text_results=True, keep_vectorise_results=False, keep_taxonomy_results=True,output_keys=None
	# When output_keys is None, the keep_* flags decide what to include.
)

runner = TrapicheWorkflowFromSequence(workflow_params=workflow_params)
result = runner.run(samples)  # sequence of dicts augmented with predictions
print(result)
runner.save("trapiche_results.ndjson")  # optional convenience save
```

Text prediction

```python
from trapiche.api import TextToBiome

ttb = TextToBiome()  # uses default model and device

texts = [x["project_description_text"] for x in samples]
text_predictions = ttb.predict(texts)
print(text_predictions)  # list[list[str]]: predicted biome labels per input text

# Optionally save last predictions
ttb.save("text_preds.json")
```

Taxonomy → community vector → biome lineage

```python
from trapiche.api import Community2vec, TaxonomyToBiome

# Vectorise one or more samples from taxonomy annotation files
c2v = Community2vec()

taxonomy_files = [x["taxonomy_files_paths"] for x in samples]

vectors = c2v.transform(taxonomy_files)

tax2b = TaxonomyToBiome()
result = tax2b.predict(community_vectors=vectors,constrain=text_predictions)
print(len(result))
print(result[0])  # pandas DataFrame with per-sample predictions

# Optional saves
c2v.save("community_vectors.npy")
tax2b.save("taxonomy_predictions.csv")
tax2b.save_vectors("taxonomy_vectors.npy")
```

## Input schema

Input record (API and CLI workflow)

One JSON object per sample in eithe NDJSON (CLI) or List (API), with the following keys:

```json
{
	"taxonomy_files_paths": ["/path/to/sample1.tsv", "/path/to/sample1_b.tsv.gz"],
	"project_description_text": "Free text describing the sample.",
	"sample_description_text": "Free text describing this specific sample variant.",
	# alternatively (if no inline text):
	# "project_description_file_path": "path/to/description.txt"
}
```

## Output schema
Output record (API and CLI workflow)
One JSON object per sample in either NDJSON (CLI) or List (API), with the following keys added to the input record:
```
 {'raw_unambiguous_prediction': ('root:Host-associated:Animal:Vertebrates:Mammals:Human:Skin',
   1.0),
  'raw_refined_prediction': {'root:Host-associated:Animal:Vertebrates:Mammals:Human:Skin': 1.0},
  'final_selected_prediction': {'root:Engineered:Food production': 1.0},
  'text_predictions': ['root:Engineered:Food production'],
  'constrained_unambiguous_prediction': ('root:Engineered:Food production',
   1.0),
  'constrained_refined_prediction': {'root:Engineered:Food production': 1.0}}
```

Best prediction is in `final_selected_prediction`.

## Sample-over-study heuristic (optional)

When enabled (via CLI flag `--sample-study-text-heuristic` or programmatically by setting `TextToBiomeParams(sample_study_text_heuristic=True)`), Trapiche will:

- Run text prediction on both `project_description_text` and `sample_description_text` when both are provided for a sample.
- Get union of the two label sets.

This heuristic can improve specificity when sample-level text refines the broader project description.


## Tests

Integration tests of API and CLI

Run tests:

```bash
python -m unittest discover -s test -p 'test_*.py' -q
```

## Data and models

Trapiche ships code only. Models used live in HugginFaceHub, and are downloaded by HF api.


## Benchmarks (placeholder)

We will release a small benchmarking section here with datasets, metrics (accuracy, F1, calibration), and comparisons against text‑only and taxonomy‑only baselines.


## Paper (placeholder)

A preprint/manuscript describing Trapiche’s joint modelling approach and training data will be linked here.


## Contributing

Issues and pull requests are welcome. Please open an issue to discuss larger changes.


## License and citation

This project is open source. See the repository metadata for licensing terms.

If you use Trapiche in your work, please cite the Trapiche paper (coming soon). A BibTeX entry will be provided here.


