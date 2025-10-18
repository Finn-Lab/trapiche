Trapiche — Joint biome classification from text and taxonomy
===========================================================

Trapiche is an open-source tool for biome classification in metagenomic studies. It combines two complementary signals:

- Text-based context: an adapted Large Language Model (LLM) predicts candidate biomes from project/sample descriptions.
- Taxonomy-based signal: a taxonomy_vectorization embedding of taxonomic profiles is fed to a feed‑forward model for deep biome lineage prediction.

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

This fetches the required taxonomy_vectorization resources, taxonomy graph, and other assets used at runtime.

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
{"project_description_file_path":"test/files/text_files/PRJEB42572_project_description.txt","taxonomy_files_paths":["test/files/taxonomy_files/ERZ19590789_FASTA_diamond.tsv.gz"]}
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
		"project_description_text": "Dental plaque microbiomes from hunter-gatherer and subsistence farmer populations in Cameroon. This study collected dental plaque samples from two non-industrial populations in Cameroon, the Baka and the Nzime. Two plaque samples were collected per individual, one from anterior teeth and one from posterior teeth.",
		"taxonomy_files_paths": [ ".prueba.bak/files/taxonomy_files/ERZ19590789_FASTA_diamond.tsv.gz"],
	},
	{
		"project_description_text": "Epipelagic bacterial communities of Canadian lakes. The NSERC Canadian LakePulse Network is a scientific initiative assessing environmental issues affecting Canadian lakes. Through multidisciplinary projects, LakePulse researchers use tools in lake science, spatial modelling, analytical chemistry, public health, and remote sensing to assess the status of over 600 lakes across various ecozones in Canada. The impacts of land-use, climate change and contaminants on lake health will be assessed to develop policies for better lake management.",
		"taxonomy_files_paths": [ ".prueba.bak/files/taxonomy_files/ERR5954428_MERGED_FASTQ_LSU_OTU.tsv",".prueba.bak/files/taxonomy_files/ERR5954428_MERGED_FASTQ_SSU_OTU.tsv"],
	},
	{
		"project_description_text": "Temporal shotgun metagenomic dissection of the coffee fermentation ecosystem. The current study employed a temporal shotgun metagenomic analysis of a prolonged (64 h) coffee fermentation process (six time points) to facilitate an in-depth dissection of the structure and functions of the coffee microbiome.",
		"taxonomy_files_paths": [ ".prueba.bak/files/taxonomy_files/ERR2231570_MERGED_FASTQ_LSU_OTU.tsv",".prueba.bak/files/taxonomy_files/ERR2231570_MERGED_FASTQ_SSU_OTU.tsv"],
	},
]

workflow_params = TrapicheWorkflowParams(  # defaults shown
	run_text=True, run_vectorise=True, run_taxonomy=True,
	keep_text_results=False, keep_vectorise_results=False, keep_taxonomy_results=False,
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

