"""Configuration for the trapiche.helpers LLM text prediction module.

All fields can be overridden via environment variables using the
``TRAPICHE_LLM_`` prefix (e.g. ``TRAPICHE_LLM_MODEL``,
``TRAPICHE_LLM_TEMPERATURE``, ``TRAPICHE_LLM_PROJECT_BATCH_SIZE``,
``TRAPICHE_LLM_SAMPLE_BATCH_SIZE``).
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMTextPredConfig(BaseSettings):
    """Parameters for LLM-assisted biome text prediction.

    Attributes:
        model: LiteLLM model string passed to ``litellm.completion()``,
            e.g. ``"openai/gpt-4o"`` or
            ``"anthropic/claude-3-5-sonnet-20241022"``.
        temperature: Sampling temperature forwarded to the LLM.
        project_batch_size: Maximum number of projects per LLM call.
            ``None`` means no project-count limit per batch.
            Env var: ``TRAPICHE_LLM_PROJECT_BATCH_SIZE``.
        sample_batch_size: Maximum number of samples per LLM call.
            ``None`` (default) means no sample-count limit per batch.
            Env var: ``TRAPICHE_LLM_SAMPLE_BATCH_SIZE``.
    """

    model_config = SettingsConfigDict(env_prefix="TRAPICHE_LLM_", case_sensitive=False)

    model: str = "openai/gpt-4o"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    project_batch_size: int | None = Field(default=5, ge=1)
    sample_batch_size: int | None = Field(default=None, ge=1)
