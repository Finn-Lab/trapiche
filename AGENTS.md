# AGENTS.md

Python package `trapiche/` (biome classification from text + taxonomy). Code-only repo: all ML models are downloaded from Hugging Face Hub at runtime (or read from a local mirror via `*_local_model_dir`).

## Project overview

Trapiche combines two prediction pathways:

1. Text descriptions are classified by a BERT multi-label model.
2. Taxonomy annotation files are converted to Community2vec embeddings and
   passed through a TensorFlow/Keras lineage classifier with KNN refinement.

## Commands

```bash
task setup            # uv sync --extra dev --extra cpu|gpu (platform-picked TF extra; TF_EXTRA=none to skip)
uv run pytest         # full test suite (README's `python -m unittest discover -s test` also works)
uv run pytest test/test_integration_api_cli.py::TestAPIIntegration::test_text_api_predict  # single test
uv run ruff check . --fix && uv run black .         # lint + format (task lint / task format)
task run -- input.ndjson [extra CLI flags]          # run trapiche CLI on a file (output: <input>_trapiche_results.ndjson)
pre-commit run --all-files
```

- Integration tests hit Hugging Face on first use and **skip gracefully** when assets/modules are unavailable — skipped tests are normal, not failures.
- CI (`.github/workflows/pre-commit-autofix.yml`) auto-commits pre-commit fixes on every push. Run `pre-commit run --all-files` locally before pushing to avoid bot fixup commits.
- Formatting: black + ruff, line-length 100. Ruff ignores E501, so black is the real enforcer.

## Environment gotchas

- Python **>=3.11, <3.13** (pyproject is authoritative).
- TensorFlow is an optional extra, imported lazily — the taxonomy deep model raises `RuntimeError` at **call time**, not import time. The text pathway works without TF. On macOS arm64 the `cpu` extra **cannot install** (tensorflow-cpu has no macOS wheels) — use the `gpu` extra (plain tensorflow) there; `task setup` already picks per-platform. `uv sync` removes unrequested extras, so never install TF ad-hoc with `uv pip install` — it gets wiped by the next `task run`.
- spaCy `en_core_sci_sm` is installed from a direct S3 URL in `dependencies` (needed only when `split_sentences=True`).
- Version comes from git tags via setuptools-scm; `trapiche/_version.py` is generated — never edit it.
- Config is pydantic-settings: workflow params via `TRAPICHE_*`, text model via `TRAPICHE_TEXT_*` (legacy un-prefixed names still accepted), vectorizer via `TRAPICHE_VECTOR_*`, taxonomy model via `TRAPICHE_TAXONOMY_*`, LLM helper via `TRAPICHE_LLM_*`. `--config` dotenv files never override exported env vars (precedence: CLI flags > env > file), and the CLI restores the environment on exit.
- Shared ontology assets (biome hierarchy, tag list) live in the **vectorizer** repo; loaders resolve them via `utils.shared_asset_params(vector_params)`. Pass `vector_params`/`vectorise_params` through new code paths so explicit `local_model_dir` overrides work offline.
- CLI writes `trapiche.log` to cwd by default (`--log-file` to change).

## Architecture notes (non-obvious)

- Pipeline: `workflow.py` (step functions `run_text_step`, `run_taxonomy_step`, orchestrated by `run_workflow`; vectorisation is called inline via `taxonomy_vectorization.vectorise_samples`) ← `api.py` (class wrappers) ← `cli.py` (NDJSON/.gz I/O). Add new steps as pure functions in `workflow.py`.
- **External text labels are the primary pathway**: if `ext_text_pred_project`/`ext_text_pred_sample` is present in **any** sample, the internal BERT classifier is skipped for the **entire batch** (batch-level short-circuit in `run_text_step`). Labels must match `root:Category[:Subcategory...]` or a `ValueError` is raised; `normalize_and_canonicalize_labels` (utils) does fuzzy fallback canonicalization. Raw external labels are preserved in output as `_raw_ext_text_pred_*`.
- In `run_workflow`, taxonomy predictions are merged into a sample's output **only if its community vector is a non-zero ndarray** — empty/unparsable taxonomy files yield `(n, 0)` vectors and silently drop taxonomy results.
- CLI output is minimal by default: keys are `TrapicheWorkflowParams.output_keys` unless `--disable-minimal-result` is passed (then `keep_*` flags decide). CLI boolean flags use `BooleanOptionalAction`: `--run-text/--no-run-text`, etc.
- `trapiche/helpers/` (extra `[helpers]` → litellm) generates `ext_text_pred_*` labels via LLM; independent of the core pipeline.
- `text_prediction.py` supports configurable probability thresholds and optional
  sentence splitting; sentence-level predictions are aggregated with `max`.
- `taxonomy_prediction.py` lazily loads the Keras model and performs batched
  prediction plus KNN refinement against the MGnify HDF5 vectors.
- `utils.py` owns Hugging Face asset resolution, taxonomy parsing, and conversion
  of NumPy/model objects into JSON-safe output.

## Design patterns

- Heavy dependencies are imported lazily where practical. Missing TensorFlow
  raises `RuntimeError` when the taxonomy model is called, not during import.
- All model assets are fetched from Hugging Face (or a local mirror selected via
  `local_model_dir`) using versioned file patterns; do not hard-code local model paths.
- Public functions and classes use Google-style docstrings. Keep workflow steps
  pure and add new orchestration steps in `workflow.py`.
- The CLI accepts NDJSON from a file or stdin and supports gzip input/output.
  Text inference uses a small configurable batch (`TextToBiomeParams.batch_size`,
  env `TRAPICHE_TEXT_BATCH_SIZE`); the taxonomy model chunks samples by
  `TRAPICHE_TAXONOMY_BATCH_SIZE` (legacy `TRAPICHE_BATCH_SIZE`). TensorFlow GPU
  memory growth is enabled by default; caps can be set with
  `TRAPICHE_TF_GPU_MEMORY_LIMIT_MB` and `TRAPICHE_TORCH_GPU_MEMORY_FRACTION`, and
  `TRAPICHE_TF_NUM_THREADS` pins TensorFlow threads (unset = framework defaults).
  CPU RAM has no hard allocator limit in either framework, so lower the batch
  sizes when needed.

## Style

- Google-style docstrings on public functions/classes; heavy deps (TF, torch, spacy) imported lazily inside functions.
