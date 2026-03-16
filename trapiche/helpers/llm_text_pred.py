"""LLM-assisted biome text prediction helper.

Wraps the GOLD ecosystem prompt template, calls any LiteLLM-supported
provider, parses and validates the result, and returns data that feeds
directly into the existing ``run_text_step`` external path via the
``ext_text_pred_project`` / ``ext_text_pred_sample`` keys.

Install the optional dependency before use::

    pip install .[helpers]
"""

from __future__ import annotations

import json
import logging
import re
from importlib.resources import files
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROMPTS = files("trapiche.helpers.prompts")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_prompt_template() -> str:
    """Read the prompt template from the package resource directory."""
    return _PROMPTS.joinpath("text_prediction_prompt.md").read_text(encoding="utf-8")


def _load_gold_taxonomy() -> str:
    """Read gold_ecosystem_tree.json from the package resource directory."""
    return _PROMPTS.joinpath("gold_ecosystem_tree.json").read_text(encoding="utf-8")


def _build_prompt(projects: list[dict]) -> str:
    """Fill placeholders in the prompt template.

    Args:
        projects: Project dicts to serialize into the prompt.

    Returns:
        str: Fully substituted prompt string.
    """
    template = _load_prompt_template()
    taxonomy = _load_gold_taxonomy()
    input_json = json.dumps(projects, indent=2, ensure_ascii=False)
    prompt = template.replace("GOLD_TAXONOMY_JSON_PLACEHOLDER", taxonomy)
    prompt = prompt.replace("INPUT_JSON_PLACEHOLDER", input_json)
    return prompt


def _parse_llm_response(text: str) -> list[dict]:
    """Extract a JSON array from an LLM response.

    Handles markdown code fences (```json ... ```) as well as bare JSON.

    Args:
        text: Raw text returned by the LLM.

    Returns:
        list[dict]: Parsed list of project objects.

    Raises:
        ValueError: If no valid JSON array can be extracted.
    """
    # Strip markdown fences if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    candidate = fence_match.group(1).strip() if fence_match else text.strip()

    # Find the outermost JSON array
    array_match = re.search(r"\[[\s\S]*\]", candidate)
    if array_match:
        candidate = array_match.group(0)

    try:
        result = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse LLM response as JSON: {exc}\nRaw text:\n{text}") from exc

    if not isinstance(result, list):
        raise ValueError(f"Expected a JSON array from LLM, got {type(result).__name__}.")

    return result


def _validate_and_clean(enriched: list[dict]) -> list[dict]:
    """Normalize, canonicalize, and drop unknown/invalid biome labels.

    Labels are first normalized (whitespace, case) and resolved against the
    GOLD ontology. Any label that is unknown or malformed is dropped with a
    warning.

    Args:
        enriched: Raw list of enriched project dicts from the LLM.

    Returns:
        list[dict]: Same structure with invalid/unknown labels removed.
    """
    from trapiche.utils import normalize_and_canonicalize_labels

    cleaned = []
    for proj in enriched:
        proj = dict(proj)
        proj_id = proj.get("project_id", "?")

        proj["_raw_project_ecosystems"] = list(proj.get("project_ecosystems", []))
        proj["project_ecosystems"] = normalize_and_canonicalize_labels(
            proj.get("project_ecosystems", []),
            warn_prefix=f"Project {proj_id} (project): ",
            fuzzy_fallback=True,
        )

        cleaned_samples = []
        for sample in proj.get("samples", []):
            sample = dict(sample)
            sid = sample.get("sample_id", "?")
            sample["_raw_sample_ecosystems"] = list(sample.get("sample_ecosystems", []))
            sample["sample_ecosystems"] = normalize_and_canonicalize_labels(
                sample.get("sample_ecosystems", []),
                warn_prefix=f"Project {proj_id} sample {sid}: ",
                fuzzy_fallback=True,
            )
            cleaned_samples.append(sample)
        proj["samples"] = cleaned_samples
        cleaned.append(proj)

    return cleaned


def _build_batches(
    projects: list[dict],
    project_batch_size: int | None,
    sample_batch_size: int | None,
) -> list[list[dict]]:
    """Split projects into batches respecting project- and sample-count limits.

    When a project has more samples than ``sample_batch_size`` allows in one
    batch, it is split across multiple batches. Each split entry repeats the
    ``project_id`` and ``PROJECT_DESCRIPTION`` so the LLM always has context.

    Args:
        projects: List of project dicts (each with a ``samples`` list).
        project_batch_size: Max projects per batch; ``None`` = unlimited.
        sample_batch_size: Max samples per batch; ``None`` = unlimited.

    Returns:
        list[list[dict]]: List of batches, each batch a list of project dicts.
    """
    batches: list[list[dict]] = []
    current_batch: list[dict] = []
    current_sample_count = 0

    def flush() -> None:
        nonlocal current_batch, current_sample_count
        if current_batch:
            batches.append(current_batch)
            current_batch = []
            current_sample_count = 0

    for project in projects:
        remaining_samples = list(project.get("samples", []))

        while True:
            # flush if project-count limit would be exceeded
            if project_batch_size is not None and len(current_batch) >= project_batch_size:
                flush()

            # flush if sample-count limit already reached
            if sample_batch_size is not None and current_sample_count >= sample_batch_size:
                flush()

            # take as many samples as the sample limit allows
            if sample_batch_size is not None:
                available = sample_batch_size - current_sample_count
                chunk, remaining_samples = remaining_samples[:available], remaining_samples[available:]
            else:
                chunk, remaining_samples = remaining_samples, []

            current_batch.append(
                {
                    "project_id": project["project_id"],
                    "PROJECT_DESCRIPTION": project["PROJECT_DESCRIPTION"],
                    "samples": chunk,
                }
            )
            current_sample_count += len(chunk)

            if not remaining_samples:
                break

    flush()
    return batches


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_biomes_from_text(
    projects: list[dict],
    model: str | None = None,
    *,
    litellm_kwargs: dict[str, Any] | None = None,
    config: "LLMTextPredConfig | None" = None,
    dry_run: bool = False,
    save_prompts_dir: str | Path | None = None,
) -> list:
    """Call an LLM to predict GOLD biome paths for each project and its samples.

    Projects are processed in batches controlled by ``config.project_batch_size``
    and ``config.sample_batch_size`` to keep individual prompts manageable.

    Args:
        projects: List of project dicts. Each dict must contain at least:
            - ``project_id`` (str)
            - ``PROJECT_DESCRIPTION`` (str)
            - ``samples`` (list of dicts with ``sample_id`` and
              ``SAMPLE_DESCRIPTION``)
        model: LiteLLM model string, e.g. ``"openai/gpt-4o"`` or
            ``"anthropic/claude-3-5-sonnet-20241022"``. When omitted,
            ``config.model`` is used.
        litellm_kwargs: Extra kwargs forwarded to ``litellm.completion()``
            (e.g. ``api_key``, ``api_base``). ``temperature`` from
            ``config.temperature`` is included automatically unless
            overridden here.
        config: :class:`~trapiche.helpers.config.LLMTextPredConfig` instance.
            When omitted a default instance is created (reads
            ``TRAPICHE_LLM_*`` env vars).
        dry_run: If True, print the prompt for the first batch to stdout and
            return an empty list without calling the LLM. Useful for
            inspecting the prompt before committing to API calls.
            Ignored when ``save_prompts_dir`` is set.
        save_prompts_dir: If set, write each batch prompt to a ``.txt`` file
            in this directory instead of calling the LLM.  The directory is
            created if it does not exist.  Returns a ``list[str]`` of saved
            file paths (one per batch).  Takes precedence over ``dry_run``.

    Returns:
        ``list[dict]`` of enriched project dicts (normal mode) **or**
        ``list[str]`` of saved prompt file paths (when ``save_prompts_dir``
        is set).  Each enriched project dict contains:
            - ``"project_ecosystems"``: list[str]
            - ``"samples"``: each sample dict enriched with
              ``"sample_ecosystems"``: list[str]

    Raises:
        ImportError: If ``litellm`` is not installed (normal mode only).
        ValueError: If the LLM response cannot be parsed.
    """
    from trapiche.helpers.config import LLMTextPredConfig

    if config is None:
        config = LLMTextPredConfig()

    batches = _build_batches(projects, config.project_batch_size, config.sample_batch_size)
    total_batches = len(batches)

    if save_prompts_dir is not None:
        save_dir = Path(save_prompts_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[str] = []
        for i, batch in enumerate(batches, start=1):
            prompt = _build_prompt(batch)
            out_path = save_dir / f"prompt_batch_{i:03d}.txt"
            out_path.write_text(prompt, encoding="utf-8")
            logger.info(
                "Saved prompt batch %d/%d → %s (%d chars)",
                i,
                total_batches,
                out_path,
                len(prompt),
            )
            saved_paths.append(str(out_path))
        return saved_paths

    if dry_run:
        first_batch = batches[0] if batches else []
        print(_build_prompt(first_batch))
        return []

    try:
        import litellm
    except ImportError as exc:
        raise ImportError(
            "litellm is required for predict_biomes_from_text. "
            "Install it with: pip install .[helpers]"
        ) from exc

    resolved_model = model if model is not None else config.model

    base_kwargs: dict[str, Any] = {"temperature": config.temperature}
    if litellm_kwargs:
        base_kwargs.update(litellm_kwargs)

    results: list[dict] = []
    for i, batch in enumerate(batches, start=1):
        prompt = _build_prompt(batch)
        logger.debug(
            "Calling LLM model=%s batch=%d/%d prompt_length=%d chars",
            resolved_model,
            i,
            total_batches,
            len(prompt),
        )
        response = litellm.completion(
            model=resolved_model,
            messages=[{"role": "user", "content": prompt}],
            **base_kwargs,
        )
        raw_text = response.choices[0].message.content
        logger.debug("LLM response length=%d chars", len(raw_text))
        enriched = _parse_llm_response(raw_text)
        enriched = _validate_and_clean(enriched)
        results.extend(enriched)

    return results


def load_enriched_from_dir(
    directory: str | Path,
    pattern: str = "*.json",
) -> list[dict]:
    """Load and merge enriched project dicts from JSON files in a directory.

    Globs ``directory`` for files matching ``pattern`` (sorted alphabetically
    for deterministic merge order), loads each as a JSON array of enriched
    project dicts, and merges entries that share the same ``project_id`` by
    extending the ``samples`` list.

    Args:
        directory: Path to the directory containing JSON files.
        pattern: Glob pattern used to select files (default: ``"*.json"``).

    Returns:
        list[dict]: Merged list of enriched project dicts, suitable for
        passing directly to :func:`to_trapiche_samples`.

    Notes:
        - Files are processed in alphabetical order.
        - When the same ``project_id`` appears in multiple files, the
          ``PROJECT_DESCRIPTION`` from the **first** file is kept; a warning
          is emitted if subsequent files carry a different description.
        - The ``samples`` lists are concatenated in file order.
    """
    directory = Path(directory)
    files_found = sorted(directory.glob(pattern))

    merged: dict[str, dict] = {}  # project_id -> project dict

    for file_path in files_found:
        with file_path.open(encoding="utf-8") as fh:
            projects = json.load(fh)
        for proj in projects:
            pid = proj["project_id"]
            if pid not in merged:
                merged[pid] = dict(proj)
                merged[pid]["samples"] = list(proj.get("samples", []))
            else:
                existing_desc = merged[pid].get("PROJECT_DESCRIPTION", "")
                new_desc = proj.get("PROJECT_DESCRIPTION", "")
                if new_desc != existing_desc:
                    logger.warning(
                        "project_id %r has differing PROJECT_DESCRIPTION in %s; keeping first.",
                        pid,
                        file_path,
                    )
                merged[pid]["samples"].extend(proj.get("samples", []))

    result = list(merged.values())
    logger.info(
        "Loaded %d projects from %d files in %s",
        len(result),
        len(files_found),
        directory,
    )
    return result


def from_workflow_samples(samples: list[dict]) -> list[dict]:
    """Convert canonical Trapiche workflow rows into predict_biomes_from_text input.

    Groups flat per-sample dicts (one dict per sample, as used in the normal
    Trapiche NDJSON workflow) into the project-grouped structure required by
    predict_biomes_from_text.

    Args:
        samples: List of per-sample dicts. Each dict must contain:
            - ``project_id`` (str): project/study identifier used for grouping.
            - ``project_description_text`` (str): free-text project description.
          Optional keys:
            - ``sample_id`` (str): if absent the row contributes no sample entry.
            - ``sample_description_text`` (str): if absent (even when sample_id
              is present) the sample is omitted from the samples list.

    Returns:
        list[dict]: Project-grouped dicts ready for predict_biomes_from_text::

            [
              {
                "project_id": "<project_id>",
                "PROJECT_DESCRIPTION": "<project_description_text of first row>",
                "samples": [
                  {"sample_id": "<sample_id>", "SAMPLE_DESCRIPTION": "<sample_description_text>"},
                  ...
                ]
              },
              ...
            ]

    Notes:
        - Rows are grouped by project_id; insertion order is preserved.
        - When multiple rows share the same project_id, the project_description_text
          from the **first** row encountered is used.
        - Rows missing sample_id are silently skipped (debug-logged).
        - Rows with sample_id but without sample_description_text are omitted
          from the samples list.

    Raises:
        KeyError: If a row is missing the ``project_id`` key.
        ValueError: If a row is missing the ``project_description_text`` key.
    """
    projects: dict[str, dict] = {}

    for row in samples:
        project_id = row["project_id"]
        if "project_description_text" not in row:
            raise ValueError(f"Row for project_id {project_id!r} is missing 'project_description_text'.")

        if project_id not in projects:
            projects[project_id] = {
                "project_id": project_id,
                "PROJECT_DESCRIPTION": row["project_description_text"],
                "samples": [],
            }

        sample_id = row.get("sample_id")
        if not sample_id:
            logger.debug("Row for project_id %r has no sample_id; skipping sample entry.", project_id)
            continue

        sample_desc = row.get("sample_description_text")
        if not sample_desc:
            continue

        projects[project_id]["samples"].append(
            {"sample_id": sample_id, "SAMPLE_DESCRIPTION": sample_desc}
        )

    return list(projects.values())


def to_trapiche_samples(
    enriched_projects: list[dict],
    base_samples: list[dict] | None = None,
) -> list[dict]:
    """Flatten enriched project dicts into per-sample dicts for Trapiche.

    Adds ``ext_text_pred_project`` and ``ext_text_pred_sample`` to every
    sample so they can be passed directly to
    ``TrapicheWorkflowFromSequence.run()``.

    Args:
        enriched_projects: Output of :func:`predict_biomes_from_text`.
        base_samples: Optional list of existing sample dicts (matched by
            ``sample_id``). When provided, the external prediction keys are
            merged into the corresponding base dict; unmatched samples are
            left as-is. When omitted, minimal dicts are built from the
            enriched structure.

    Returns:
        list[dict]: Per-sample dicts with ``ext_text_pred_project`` and
        ``ext_text_pred_sample`` populated.
    """
    from trapiche.utils import normalize_and_canonicalize_labels

    # Build lookup from enriched structure
    # Maps sample_id -> (project_ecosystems, sample_ecosystems, raw_proj, raw_samp)
    pred_by_sample: dict[str, tuple[list[str], list[str], list[str], list[str]]] = {}
    for proj in enriched_projects:
        proj_id = proj.get("project_id", "?")
        raw_proj = list(proj.get("_raw_project_ecosystems", proj.get("project_ecosystems", [])))
        proj_labels = normalize_and_canonicalize_labels(
            proj.get("project_ecosystems", []),
            warn_prefix=f"Project {proj_id} (project): ",
            fuzzy_fallback=True,
        )
        for sample in proj.get("samples", []):
            sid = sample.get("sample_id")
            raw_samp = list(
                sample.get("_raw_sample_ecosystems", sample.get("sample_ecosystems", []))
            )
            samp_labels = normalize_and_canonicalize_labels(
                sample.get("sample_ecosystems", []),
                warn_prefix=f"Project {proj_id} sample {sid}: ",
                fuzzy_fallback=True,
            )
            if sid is not None:
                pred_by_sample[sid] = (proj_labels, samp_labels, raw_proj, raw_samp)

    if base_samples is not None:
        result = []
        for base in base_samples:
            sample = dict(base)
            sid = sample.get("sample_id")
            if sid in pred_by_sample:
                proj_labels, samp_labels, raw_proj, raw_samp = pred_by_sample[sid]
                sample["ext_text_pred_project"] = proj_labels
                sample["ext_text_pred_sample"] = samp_labels
                sample["_raw_ext_text_pred_project"] = raw_proj
                sample["_raw_ext_text_pred_sample"] = raw_samp
            result.append(sample)
        return result

    # Build minimal dicts from the enriched structure
    result = []
    for proj in enriched_projects:
        proj_id = proj.get("project_id", "?")
        raw_proj = list(proj.get("_raw_project_ecosystems", proj.get("project_ecosystems", [])))
        proj_labels = normalize_and_canonicalize_labels(
            proj.get("project_ecosystems", []),
            warn_prefix=f"Project {proj_id} (project): ",
            fuzzy_fallback=True,
        )
        for sample in proj.get("samples", []):
            sid = sample.get("sample_id")
            raw_samp = list(
                sample.get("_raw_sample_ecosystems", sample.get("sample_ecosystems", []))
            )
            samp_labels = normalize_and_canonicalize_labels(
                sample.get("sample_ecosystems", []),
                warn_prefix=f"Project {proj_id} sample {sid}: ",
                fuzzy_fallback=True,
            )
            sample_dict: dict[str, Any] = {}
            if sid is not None:
                sample_dict["sample_id"] = sid
            sample_dict["ext_text_pred_project"] = proj_labels
            sample_dict["ext_text_pred_sample"] = samp_labels
            sample_dict["_raw_ext_text_pred_project"] = raw_proj
            sample_dict["_raw_ext_text_pred_sample"] = raw_samp
            result.append(sample_dict)
    return result
