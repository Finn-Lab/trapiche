# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

**Trapiche** is a Python biome classification toolkit that combines two complementary ML pathways to predict environmental biomes from metagenomic data:
1. **Text-based**: Free-form sample/project descriptions → BERT multi-label classifier → biome labels
2. **Taxonomy-based**: Taxonomic annotation files → Community2vec embedding → deep Keras model + KNN refinement → lineage predictions

## Commands

### Installation
```bash
pip install .           # text pathway only
pip install .[cpu]      # + TensorFlow CPU (enables taxonomy deep model)
pip install .[gpu]      # + TensorFlow GPU
pip install .[dev]      # + pre-commit, ruff, black
```

### Linting and formatting
```bash
black trapiche/         # format
ruff check trapiche/    # lint
pre-commit run --all-files  # run all hooks
```

### Tests
```bash
python -m unittest discover -s test -p 'test_*.py' -q
python -m unittest test.test_integration_api_cli.ClassName.test_method  # single test
```

### CLI
```bash
trapiche input.ndjson                          # run full workflow
trapiche input.ndjson --no-taxonomy            # text-only
trapiche - < input.ndjson                      # stdin
trapiche input.ndjson --disable-minimal-result # full output keys
```

## Architecture

### Module responsibilities

- **`trapiche/config.py`** — Pydantic `BaseSettings` dataclasses (`TrapicheWorkflowParams`, `TextToBiomeParams`, `TaxonomyToVectorParams`, `TaxonomyToBiomeParams`). Settings can be overridden via `TRAPICHE_*` env vars.
- **`trapiche/workflow.py`** — Pure functions orchestrating the three steps: `run_text_step`, `run_vectorise_step`, `run_taxonomy_step`, `run_workflow`. New steps should be added as pure functions here.
- **`trapiche/api.py`** — High-level Python API: `TextToBiome`, `Community2vec`, `TaxonomyToBiome`, `TrapicheWorkflowFromSequence`.
- **`trapiche/cli.py`** — NDJSON I/O (supports `.gz`), maps CLI flags to `TrapicheWorkflowParams`.
- **`trapiche/text_prediction.py`** — BERT multi-label classifier; configurable threshold (`0.01`, `max`, `top-N`); optional sentence splitting via spaCy/NLTK; aggregation = max.
- **`trapiche/taxonomy_vectorization.py`** — Community2vec over taxonomy subgraphs from annotation files; returns `(n, dim)` or `(n, 0)` for empty/unparsable inputs.
- **`trapiche/taxonomy_prediction.py`** — Loads Keras model lazily; `from_probs_to_pred` maps logits to candidates with optional text-constraints (prefix matching); `refine_predictions_knn_batch` refines with KNN against MGnify HDF5 vectors.
- **`trapiche/utils.py`** — HF model retrieval (`_get_hf_model_path` with versioned glob patterns), taxonomy file parsing (`tax_annotations_from_file`), `obj_to_serializable`.

### Key design patterns

- **Lazy imports**: TensorFlow and other heavy libs are imported only when actually called. Missing TF raises `RuntimeError` at use time, not import time.
- **External assets**: All models hosted on Hugging Face Hub (`SantiagoSanchezF/trapiche-biome-*`). Fetched by versioned glob patterns like `*_v{version}.json`.
- **Vectorize step input normalization**: `run_vectorise_step` accepts path, dir, list, or list[list] — normalize at the workflow boundary.
- **Study-level heuristic**: When `--sample-study-text-heuristic` is enabled, `run_text_step` unions project+sample text labels and emits `text_predictions_project`/`text_predictions_sample` keys.
- **Taxonomy prediction outputs** (per sample): `raw_top_predictions`, `raw_unambiguous_prediction`, `raw_refined_prediction`, `constrained_*` variants, and `final_selected_prediction`.
- **Minimal result**: CLI defaults to compact output; `--disable-minimal-result` lets `TrapicheWorkflowParams.output_keys` control which keys survive.

## Code style

- Google-style docstrings for all public functions/classes; keep lines ≤80 chars.
- Black (line-length=100) + Ruff for formatting/linting (checks: E, F, I, UP, B, SIM, C4).
- When adding new workflow steps: keep functions pure in `workflow.py`, import heavy libs lazily, extend config dataclasses and CLI flags accordingly.
